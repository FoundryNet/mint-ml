use anchor_lang::prelude::*;
use anchor_spl::token::{self, Mint, MintTo, Token, TokenAccount};

declare_id!("AyFBC6DBStSbrau3wfFZzsX5rX14nx8Gkp8TqF687F5X");

const BASE_RATE_MICRO: u64 = 5000;
const MICRO_MINT: u64 = 1_000_000;
const TRUST_START: u8 = 100;
const WARMUP_JOBS: u32 = 30;
const MIN_COMPLEXITY_MULTIPLIER: u32 = 500;
const MAX_COMPLEXITY_MULTIPLIER: u32 = 2000;
const COMPLEXITY_SCALE: u32 = 1000;

#[program]
pub mod foundry_net {
    use super::*;

    pub fn initialize(
        ctx: Context<Initialize>, 
        oracle: Pubkey,
        mint: Pubkey,
        genesis_supply: u64,
    ) -> Result<()> {
        let state = &mut ctx.accounts.state;
        state.authority = ctx.accounts.authority.key();
        state.oracle = oracle;
        state.mint = mint;
        state.genesis_supply = genesis_supply;
        state.total_mint_minted = 0;
        state.total_jobs = 0;
        state.total_machines = 0;
        state.total_duration_seconds = 0;
        state.total_complexity_sum = 0;
        state.initialized_at = Clock::get()?.unix_timestamp;
        state.window_jobs = 0;
        state.window_duration = 0;
        state.window_complexity_sum = 0;
        state.window_start = Clock::get()?.unix_timestamp;
        
        msg!("FoundryNet initialized. Oracle: {}. Mint: {}. Genesis: {} MINT", 
            oracle, mint, genesis_supply / MICRO_MINT);
        Ok(())
    }

    pub fn register_machine(ctx: Context<RegisterMachine>) -> Result<()> {
        let state = &mut ctx.accounts.state;
        let machine_state = &mut ctx.accounts.machine_state;
        
        machine_state.machine = ctx.accounts.machine.key();
        machine_state.owner = ctx.accounts.owner.key();
        machine_state.trust_score = TRUST_START;
        machine_state.job_count = 0;
        machine_state.total_earned_micro = 0;
        machine_state.total_duration = 0;
        machine_state.complexity_sum = 0;
        machine_state.is_banned = false;
        machine_state.on_probation = false;
        machine_state.probation_count = 0;
        machine_state.probation_started_at = 0;
        machine_state.registered_at = Clock::get()?.unix_timestamp;
        machine_state.last_job_at = 0;
        
        state.total_machines += 1;
        
        msg!("Machine registered: {}. Owner: {}. Trust: {}", 
            ctx.accounts.machine.key(), ctx.accounts.owner.key(), TRUST_START);
        Ok(())
    }

    pub fn record_job(
        ctx: Context<RecordJob>,
        job_hash: String,
        duration_seconds: u64,
        complexity_claimed: u32,
    ) -> Result<()> {
        let state = &mut ctx.accounts.state;
        let machine_state = &mut ctx.accounts.machine_state;
        let job = &mut ctx.accounts.job;
        let now = Clock::get()?.unix_timestamp;
        
        require!(!machine_state.is_banned, ErrorCode::MachineBanned);
        require!(duration_seconds > 0, ErrorCode::InvalidDuration);
        require!(complexity_claimed >= 500 && complexity_claimed <= 2000, ErrorCode::InvalidComplexity);
        
        let network_avg_complexity = calculate_network_avg_complexity(state);
        let normalized_complexity = normalize_complexity(complexity_claimed, network_avg_complexity);
        let warmup = calculate_warmup(machine_state.job_count);
        
        let base_reward_micro = (duration_seconds as u128)
            .checked_mul(BASE_RATE_MICRO as u128).unwrap()
            .checked_mul(normalized_complexity as u128).unwrap()
            .checked_mul(warmup as u128).unwrap()
            .checked_div((COMPLEXITY_SCALE as u128) * (COMPLEXITY_SCALE as u128)).unwrap() as u64;
        
        job.job_hash = job_hash.clone();
        job.machine = ctx.accounts.machine.key();
        job.duration_seconds = duration_seconds;
        job.complexity_claimed = complexity_claimed;
        job.normalized_complexity = normalized_complexity;
        job.base_reward_micro = base_reward_micro;
        job.timestamp = now;
        job.ml_confidence = 0;
        job.trust_delta = 0;
        job.final_reward_micro = 0;
        job.settled = false;
        job.oracle_scored = false;
        
        machine_state.job_count += 1;
        machine_state.total_duration += duration_seconds;
        machine_state.complexity_sum += complexity_claimed as u64;
        machine_state.last_job_at = now;
        
        state.total_jobs += 1;
        state.total_duration_seconds += duration_seconds;
        state.total_complexity_sum += complexity_claimed as u64;
        
        maybe_rotate_window(state, now);
        state.window_jobs += 1;
        state.window_duration += duration_seconds;
        state.window_complexity_sum += complexity_claimed as u64;
        
        msg!("Job recorded: {} | {}s @ {} | base: {} micro-MINT", 
            job_hash, duration_seconds, complexity_claimed, base_reward_micro);
        Ok(())
    }

    pub fn update_trust(
        ctx: Context<UpdateTrust>,
        job_hash: String,
        ml_confidence: u32,
        trust_delta: i8,
    ) -> Result<()> {
        let machine_state = &mut ctx.accounts.machine_state;
        let job = &mut ctx.accounts.job;
        let now = Clock::get()?.unix_timestamp;
        
        require!(job.job_hash == job_hash, ErrorCode::JobHashMismatch);
        require!(!job.oracle_scored, ErrorCode::AlreadyScored);
        
        job.ml_confidence = ml_confidence;
        job.trust_delta = trust_delta;
        job.oracle_scored = true;
        
        let old_trust = machine_state.trust_score;
        let new_trust = (machine_state.trust_score as i16 + trust_delta as i16)
            .max(0)
            .min(100) as u8;
        machine_state.trust_score = new_trust;
        
        if new_trust == 0 {
            if !machine_state.on_probation {
                machine_state.on_probation = true;
                machine_state.probation_count = machine_state.probation_count.saturating_add(1);
                machine_state.probation_started_at = now;
                msg!("PROBATION: {} (count: {})", machine_state.machine, machine_state.probation_count);
            } else {
                machine_state.is_banned = true;
                msg!("BANNED: {} (repeat zero-trust)", machine_state.machine);
            }
        } else if machine_state.on_probation {
            machine_state.on_probation = false;
            machine_state.probation_started_at = 0;
            msg!("RECOVERED: {} (trust: {})", machine_state.machine, new_trust);
        }
        
        msg!("Trust: {} -> {} (delta: {}, ML: {}%)", old_trust, new_trust, trust_delta, ml_confidence / 10);
        Ok(())
    }

    pub fn settle_job(ctx: Context<SettleJob>, mint_authority_bump: u8) -> Result<()> {
        let state = &mut ctx.accounts.state;
        let machine_state = &mut ctx.accounts.machine_state;
        let job = &mut ctx.accounts.job;
        let now = Clock::get()?.unix_timestamp;
        
        require!(job.oracle_scored, ErrorCode::NotScored);
        require!(!job.settled, ErrorCode::AlreadySettled);
        require!(!machine_state.is_banned, ErrorCode::MachineBanned);
        
        if machine_state.on_probation {
            job.final_reward_micro = 0;
            job.settled = true;
            machine_state.last_job_at = now;
            msg!("Settled (PROBATION): {} | reward: 0", job.job_hash);
            return Ok(());
        }
        
        let age_seconds = now - job.timestamp;
        let decay = calculate_decay(age_seconds);
        let trust_multiplier = machine_state.trust_score as u32 * 10;
        
        let final_reward_micro = (job.base_reward_micro as u128)
            .checked_mul(trust_multiplier as u128).unwrap()
            .checked_mul(decay as u128).unwrap()
            .checked_div((COMPLEXITY_SCALE as u128) * (COMPLEXITY_SCALE as u128)).unwrap() as u64;
        
        job.final_reward_micro = final_reward_micro;
        job.settled = true;
        machine_state.total_earned_micro += final_reward_micro;
        machine_state.last_job_at = now;
        state.total_mint_minted += final_reward_micro;
        
        if final_reward_micro > 0 {
            let seeds = &[b"mint_authority".as_ref(), &[mint_authority_bump]];
            let signer_seeds = &[&seeds[..]];
            
            token::mint_to(
                CpiContext::new_with_signer(
                    ctx.accounts.token_program.to_account_info(),
                    MintTo {
                        mint: ctx.accounts.mint.to_account_info(),
                        to: ctx.accounts.machine_token_account.to_account_info(),
                        authority: ctx.accounts.mint_authority.to_account_info(),
                    },
                    signer_seeds,
                ),
                final_reward_micro,
            )?;
        }
        
        msg!("Settled: {} | {} MINT | trust: {} | supply: {} MINT",
            job.job_hash, final_reward_micro / MICRO_MINT, machine_state.trust_score,
            (state.genesis_supply + state.total_mint_minted) / MICRO_MINT);
        Ok(())
    }

    pub fn update_oracle(ctx: Context<UpdateOracle>, new_oracle: Pubkey) -> Result<()> {
        let state = &mut ctx.accounts.state;
        state.oracle = new_oracle;
        msg!("Oracle updated: {}", new_oracle);
        Ok(())
    }
}

fn calculate_network_avg_complexity(state: &NetworkState) -> u32 {
    if state.window_jobs == 0 { return COMPLEXITY_SCALE; }
    let avg = state.window_complexity_sum / state.window_jobs;
    avg.max(500).min(2000) as u32
}

fn normalize_complexity(claimed: u32, network_avg: u32) -> u32 {
    let normalized = (claimed as u64 * COMPLEXITY_SCALE as u64) / network_avg as u64;
    normalized.max(MIN_COMPLEXITY_MULTIPLIER as u64).min(MAX_COMPLEXITY_MULTIPLIER as u64) as u32
}

fn calculate_warmup(job_count: u32) -> u32 {
    let progress = (job_count * COMPLEXITY_SCALE / WARMUP_JOBS).min(COMPLEXITY_SCALE);
    500 + (500 * progress / COMPLEXITY_SCALE)
}

fn calculate_decay(age_seconds: i64) -> u32 {
    if age_seconds <= 0 { return COMPLEXITY_SCALE; }
    let age_days = age_seconds / (24 * 60 * 60);
    match age_days {
        0 => 1000, 1 => 870, 2 => 758, 3 => 660, 4 => 574,
        5 => 500, 6 => 435, 7 => 379, 8 => 330, 9 => 287, 10 => 250,
        _ => 100.max(250 >> ((age_days - 10) / 5) as u32),
    }
}

fn maybe_rotate_window(state: &mut NetworkState, now: i64) {
    let window_age = now - state.window_start;
    if window_age > 7 * 24 * 60 * 60 {
        state.window_jobs = 0;
        state.window_duration = 0;
        state.window_complexity_sum = 0;
        state.window_start = now;
    }
}

#[account]
pub struct NetworkState {
    pub authority: Pubkey,
    pub oracle: Pubkey,
    pub mint: Pubkey,
    pub genesis_supply: u64,
    pub total_mint_minted: u64,
    pub total_jobs: u64,
    pub total_machines: u32,
    pub total_duration_seconds: u64,
    pub total_complexity_sum: u64,
    pub initialized_at: i64,
    pub window_jobs: u64,
    pub window_duration: u64,
    pub window_complexity_sum: u64,
    pub window_start: i64,
}

#[account]
pub struct MachineState {
    pub machine: Pubkey,
    pub owner: Pubkey,
    pub trust_score: u8,
    pub job_count: u32,
    pub total_earned_micro: u64,
    pub total_duration: u64,
    pub complexity_sum: u64,
    pub is_banned: bool,
    pub on_probation: bool,
    pub probation_count: u8,
    pub probation_started_at: i64,
    pub registered_at: i64,
    pub last_job_at: i64,
}

#[account]
pub struct Job {
    pub job_hash: String,
    pub machine: Pubkey,
    pub duration_seconds: u64,
    pub complexity_claimed: u32,
    pub normalized_complexity: u32,
    pub base_reward_micro: u64,
    pub final_reward_micro: u64,
    pub timestamp: i64,
    pub ml_confidence: u32,
    pub trust_delta: i8,
    pub settled: bool,
    pub oracle_scored: bool,
}

#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(init, payer = authority, space = 8 + 32 + 32 + 32 + 8 + 8 + 8 + 4 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 64)]
    pub state: Account<'info, NetworkState>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct RegisterMachine<'info> {
    #[account(mut)]
    pub state: Account<'info, NetworkState>,
    #[account(init, payer = payer, space = 8 + 32 + 32 + 1 + 4 + 8 + 8 + 8 + 1 + 1 + 1 + 8 + 8 + 8 + 64,
        seeds = [b"machine", machine.key().as_ref()], bump)]
    pub machine_state: Account<'info, MachineState>,
    pub machine: Signer<'info>,
    /// CHECK: Owner of machine's token account
    pub owner: AccountInfo<'info>,
    #[account(mut)]
    pub payer: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(job_hash: String)]
pub struct RecordJob<'info> {
    #[account(mut)]
    pub state: Account<'info, NetworkState>,
    #[account(mut, seeds = [b"machine", machine.key().as_ref()], bump,
        constraint = machine_state.machine == machine.key())]
    pub machine_state: Account<'info, MachineState>,
    #[account(init, payer = payer, space = 8 + 64 + 32 + 8 + 4 + 4 + 8 + 8 + 8 + 4 + 1 + 1 + 1 + 64,
        seeds = [b"job", job_hash.as_bytes()], bump)]
    pub job: Account<'info, Job>,
    pub machine: Signer<'info>,
    #[account(mut)]
    pub payer: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(job_hash: String)]
pub struct UpdateTrust<'info> {
    #[account(constraint = state.oracle == oracle.key() @ ErrorCode::UnauthorizedOracle)]
    pub state: Account<'info, NetworkState>,
    #[account(mut, seeds = [b"machine", job.machine.as_ref()], bump)]
    pub machine_state: Account<'info, MachineState>,
    #[account(mut, seeds = [b"job", job_hash.as_bytes()], bump)]
    pub job: Account<'info, Job>,
    pub oracle: Signer<'info>,
}

#[derive(Accounts)]
pub struct SettleJob<'info> {
    #[account(mut)]
    pub state: Account<'info, NetworkState>,
    #[account(mut, seeds = [b"machine", job.machine.as_ref()], bump)]
    pub machine_state: Account<'info, MachineState>,
    #[account(mut)]
    pub job: Account<'info, Job>,
    #[account(mut, constraint = mint.key() == state.mint @ ErrorCode::InvalidMint)]
    pub mint: Account<'info, Mint>,
    #[account(mut, constraint = machine_token_account.mint == mint.key(),
        constraint = machine_token_account.owner == machine_state.owner)]
    pub machine_token_account: Account<'info, TokenAccount>,
    /// CHECK: PDA mint authority
    #[account(seeds = [b"mint_authority"], bump)]
    pub mint_authority: AccountInfo<'info>,
    pub token_program: Program<'info, Token>,
    pub settler: Signer<'info>,
}

#[derive(Accounts)]
pub struct UpdateOracle<'info> {
    #[account(mut, constraint = state.authority == authority.key() @ ErrorCode::Unauthorized)]
    pub state: Account<'info, NetworkState>,
    pub authority: Signer<'info>,
}

#[error_code]
pub enum ErrorCode {
    #[msg("Machine is banned")]
    MachineBanned,
    #[msg("Invalid duration")]
    InvalidDuration,
    #[msg("Invalid complexity (must be 500-2000)")]
    InvalidComplexity,
    #[msg("Job hash mismatch")]
    JobHashMismatch,
    #[msg("Job already scored")]
    AlreadyScored,
    #[msg("Job not scored")]
    NotScored,
    #[msg("Job already settled")]
    AlreadySettled,
    #[msg("Unauthorized oracle")]
    UnauthorizedOracle,
    #[msg("Unauthorized")]
    Unauthorized,
    #[msg("Invalid mint")]
    InvalidMint,
}
