from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import joblib
import json
import numpy as np
from datetime import datetime
from collections import defaultdict
import base58
import struct
import os
import time
import csv
import io
import math

app = Flask(__name__)
CORS(app)

model = joblib.load('model.pkl')
with open('model_metadata.json', 'r') as f:
    config = json.load(f)
with open('model_features.json', 'r') as f:
    feature_list = json.load(f)

print(f"Model loaded. {len(feature_list)} features.")

# ============ Configuration ============

PROGRAM_ID = os.environ.get('PROGRAM_ID', '4ZvTZ3skfeMF3ZGyABoazPa9tiudw2QSwuVKn45t2AKL')
STATE_ACCOUNT = os.environ.get('STATE_ACCOUNT', '2Lm7hrtqK9W5tykVu4U37nUNJiiFh6WQ1rD8ZJWXomr2')
SOLANA_RPC = os.environ.get('SOLANA_RPC', 'https://mainnet.helius-rpc.com/?api-key=2c13462d-4a64-4c5b-b410-1520219d73aa')
ORACLE_PRIVATE_KEY = os.environ.get('ORACLE_PRIVATE_KEY', '')

# Token addresses (mainnet)
MINT_TOKEN = os.environ.get('MINT_TOKEN', '5Pd4YBgFdih88vAFGAEEsk2JpixrZDJpRynTWvqPy5da')
GENESIS_TREASURY_ATA = os.environ.get('GENESIS_TREASURY_ATA', 'JYkvEAiSmPTXMp1KDmgk9LLZVgNRU7oxXEw3L7veu2z')
PERSONAL_FEE_ATA = os.environ.get('PERSONAL_FEE_ATA', 'CWWXT7dkMrYCraZqffgG1Fk87ZWhqNGEznLfg9B5eRmU')
PROTOCOL_FEE_ATA = os.environ.get('PROTOCOL_FEE_ATA', 'Hd1usKUanHb5zjryZrr3iGujFJHq4Tcg3Frpsrejq2L5')

# SPL Token Program
TOKEN_PROGRAM_ID = 'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA'
ASSOCIATED_TOKEN_PROGRAM_ID = 'ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL'

oracle_keypair = None
if ORACLE_PRIVATE_KEY:
    try:
        key_bytes = json.loads(ORACLE_PRIVATE_KEY)
        oracle_keypair = bytes(key_bytes)
        print(f"Oracle keypair loaded: {len(oracle_keypair)} bytes")
    except Exception as e:
        print(f"Failed to load oracle keypair: {e}")

# ============ Economic Constants ============

BASE_RATE = 0.005  # MINT per second
WARMUP_JOBS = 30
DECAY_HALFLIFE_DAYS = 5
COMPLEXITY_WINDOW_DAYS = 7
MIN_COMPLEXITY_MULT = 0.5
MAX_COMPLEXITY_MULT = 2.0

# ============ State Tracking ============

machine_history = defaultdict(lambda: {
    'job_count': 0,
    'total_earned': 0.0,
    'total_duration': 0,
    'complexities': [],
    'durations': [],
    'timestamps': [],
    'trust_score': 100
})

network_stats = {
    'complexities': [],
    'durations': [],
    'timestamps': [],
    'total_jobs': 0,
    'total_duration': 0,
    'total_mint_minted': 0.0
}

scored_jobs = []
community_disputes = []

# ============ Economic Functions ============

def get_network_avg_complexity():
    if not network_stats['complexities']:
        return 1.0
    now = time.time()
    window_start = now - (COMPLEXITY_WINDOW_DAYS * 24 * 60 * 60)
    recent = [
        c for c, t in zip(network_stats['complexities'][-5000:], network_stats['timestamps'][-5000:])
        if t > window_start
    ]
    if not recent:
        return 1.0
    return np.mean(recent)

def normalize_complexity(claimed, network_avg):
    if network_avg <= 0:
        network_avg = 1.0
    normalized = claimed / network_avg
    return max(MIN_COMPLEXITY_MULT, min(MAX_COMPLEXITY_MULT, normalized))

def calculate_warmup(job_count):
    return 0.5 + 0.5 * min(1.0, job_count / WARMUP_JOBS)

def calculate_decay(age_days):
    if age_days <= 0:
        return 1.0
    return math.exp(-0.693 * age_days / DECAY_HALFLIFE_DAYS)

def calculate_base_reward(duration_seconds, complexity, job_count):
    network_avg = get_network_avg_complexity()
    normalized = normalize_complexity(complexity, network_avg)
    warmup = calculate_warmup(job_count)
    base = duration_seconds * BASE_RATE * normalized * warmup
    return {
        'base_reward': round(base, 6),
        'duration_seconds': duration_seconds,
        'complexity_claimed': complexity,
        'network_avg_complexity': round(network_avg, 3),
        'normalized_complexity': round(normalized, 3),
        'warmup_multiplier': round(warmup, 3),
        'base_rate': BASE_RATE
    }

def calculate_final_reward(base_reward, trust_score, age_days=0):
    trust_mult = trust_score / 100.0
    decay_mult = calculate_decay(age_days)
    final = base_reward * trust_mult * decay_mult
    return {
        'final_reward': round(final, 6),
        'base_reward': base_reward,
        'trust_multiplier': round(trust_mult, 3),
        'decay_multiplier': round(decay_mult, 3)
    }

# ============ ML Functions ============

def get_network_stats_for_ml():
    if not network_stats['complexities']:
        return {'avg_c': 1.0, 'std_c': 0.3, 'avg_d': 500, 'std_d': 300}
    c = network_stats['complexities'][-1000:]
    d = network_stats['durations'][-1000:]
    return {
        'avg_c': np.mean(c),
        'std_c': np.std(c) or 0.3,
        'avg_d': np.mean(d),
        'std_d': np.std(d) or 300
    }

def get_machine_stats(mid):
    h = machine_history[mid]
    net = get_network_stats_for_ml()
    if h['job_count'] < 10:
        return net
    c, d = h['complexities'][-100:], h['durations'][-100:]
    return {
        'avg_c': np.mean(c),
        'std_c': np.std(c) or net['std_c'],
        'avg_d': np.mean(d),
        'std_d': np.std(d) or net['std_d']
    }

def build_features(data):
    mid = data.get('machine_id', 'unknown')
    c = float(data.get('complexity_claimed', 1.0))
    dur = float(data.get('duration_seconds', 300))
    rew = float(data.get('reward_gross', 5.0))
    net, mach = get_network_stats_for_ml(), get_machine_stats(mid)
    h = machine_history[mid]
    warmup = calculate_warmup(h['job_count'])
    return {
        'complexity_claimed': c, 'duration_seconds': dur, 'duration_minutes': dur/60,
        'reward_gross': rew, 'reward_per_second': rew/(dur+1), 'reward_per_complexity': rew/(c+0.1),
        'network_avg_complexity': net['avg_c'], 'network_avg_duration': net['avg_d'],
        'network_complexity_std': net['std_c'], 'network_duration_std': net['std_d'],
        'jobs_in_window': data.get('jobs_in_window', 100),
        'days_since_launch': data.get('days_since_launch', 1.0),
        'activity_ratio': 1.0,
        'decay_multiplier': data.get('decay_multiplier', 1.0),
        'warmup_multiplier': warmup,
        'machine_job_count': h['job_count'],
        'machine_total_earned': h['total_earned'],
        'machine_avg_complexity': mach['avg_c'],
        'machine_avg_duration': mach['avg_d'],
        'machine_complexity_std': mach['std_c'],
        'machine_duration_std': mach['std_d'],
        'complexity_vs_network': c - net['avg_c'],
        'complexity_vs_machine': c - mach['avg_c'],
        'complexity_zscore_network': (c - net['avg_c']) / (net['std_c'] + 0.01),
        'complexity_zscore_machine': (c - mach['avg_c']) / (mach['std_c'] + 0.01),
        'duration_vs_network': dur - net['avg_d'],
        'duration_vs_machine': dur - mach['avg_d'],
        'duration_zscore_network': (dur - net['avg_d']) / (net['std_d'] + 1),
        'duration_zscore_machine': (dur - mach['avg_d']) / (mach['std_d'] + 1),
        'time_since_last_job': data.get('time_since_last_job', 0),
        'jobs_last_hour_machine': data.get('jobs_last_hour_machine', 0),
        'is_new_machine': 1 if h['job_count'] < 10 else 0,
        'complexity_duration_ratio': c / (dur/60 + 0.1),
        'reward_efficiency': rew / (dur + 1),
        'earning_rate': h['total_earned'] / (h['job_count'] + 1),
    }

def update_history(mid, c, dur, reward):
    now = time.time()
    h = machine_history[mid]
    h['job_count'] += 1
    h['total_earned'] += reward
    h['total_duration'] += dur
    h['complexities'].append(c)
    h['durations'].append(dur)
    h['timestamps'].append(now)
    if len(h['complexities']) > 500:
        h['complexities'] = h['complexities'][-500:]
        h['durations'] = h['durations'][-500:]
        h['timestamps'] = h['timestamps'][-500:]
    network_stats['complexities'].append(c)
    network_stats['durations'].append(dur)
    network_stats['timestamps'].append(now)
    network_stats['total_jobs'] += 1
    network_stats['total_duration'] += dur
    network_stats['total_mint_minted'] += reward
    if len(network_stats['complexities']) > 5000:
        network_stats['complexities'] = network_stats['complexities'][-5000:]
        network_stats['durations'] = network_stats['durations'][-5000:]
        network_stats['timestamps'] = network_stats['timestamps'][-5000:]

def score_job(data):
    mid = data.get('machine_id', 'unknown')
    dur = float(data.get('duration_seconds', 300))
    complexity = float(data.get('complexity_claimed', 1.0))
    h = machine_history[mid]
    features = build_features(data)
    vec = [features.get(f, 0) for f in feature_list]
    confidence = float(model.predict_proba([vec])[0][1])
    if confidence > 0.70:
        trust_delta, action = -5, 'flag_strong'
    elif confidence > 0.50:
        trust_delta, action = -2, 'flag_soft'
    elif confidence < 0.25:
        trust_delta, action = 1, 'clean'
    else:
        trust_delta, action = 0, 'neutral'
    old_trust = h['trust_score']
    h['trust_score'] = max(0, min(100, old_trust + trust_delta))
    base_result = calculate_base_reward(dur, complexity, h['job_count'])
    final_result = calculate_final_reward(base_result['base_reward'], h['trust_score'], age_days=0)
    update_history(mid, complexity, dur, final_result['final_reward'])
    return {
        'confidence': confidence,
        'trust_delta': trust_delta,
        'action': action,
        'economics': {
            'base_rate': BASE_RATE,
            'duration_seconds': dur,
            'complexity_claimed': complexity,
            'network_avg_complexity': base_result['network_avg_complexity'],
            'normalized_complexity': base_result['normalized_complexity'],
            'warmup_multiplier': base_result['warmup_multiplier'],
            'base_reward': base_result['base_reward'],
            'trust_score': h['trust_score'],
            'trust_multiplier': final_result['trust_multiplier'],
            'decay_multiplier': final_result['decay_multiplier'],
            'final_reward': final_result['final_reward'],
            'machine_jobs': h['job_count'],
            'network_jobs': network_stats['total_jobs'],
            'total_mint_minted': round(network_stats['total_mint_minted'], 2)
        }
    }

# ============ Solana Integration ============

def decode_record_job_instruction(data_b58):
    try:
        data = base58.b58decode(data_b58)
        offset = 8
        if len(data) > offset + 4:
            str_len = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            job_hash = data[offset:offset+str_len].decode('utf-8', errors='ignore')
            offset += str_len
            if len(data) >= offset + 8:
                duration_sec = struct.unpack('<Q', data[offset:offset+8])[0]
                offset += 8
                if len(data) >= offset + 4:
                    complexity_fixed = struct.unpack('<I', data[offset:offset+4])[0]
                    return {
                        'job_hash': job_hash,
                        'duration_seconds': duration_sec,
                        'complexity_claimed': complexity_fixed / 1000.0
                    }
        return None
    except:
        return None

def get_associated_token_address(owner_pubkey, mint_pubkey):
    """Derive the associated token address for an owner and mint"""
    from solders.pubkey import Pubkey
    owner = Pubkey.from_string(owner_pubkey) if isinstance(owner_pubkey, str) else owner_pubkey
    mint = Pubkey.from_string(mint_pubkey) if isinstance(mint_pubkey, str) else mint_pubkey
    token_program = Pubkey.from_string(TOKEN_PROGRAM_ID)
    ata_program = Pubkey.from_string(ASSOCIATED_TOKEN_PROGRAM_ID)
    
    # ATA is a PDA derived from [owner, token_program, mint]
    ata, _ = Pubkey.find_program_address(
        [bytes(owner), bytes(token_program), bytes(mint)],
        ata_program
    )
    return ata

def call_update_trust(machine_pubkey, job_pubkey, job_hash, ml_confidence, trust_delta):
    try:
        if not oracle_keypair or not STATE_ACCOUNT:
            return {'status': 'skipped', 'reason': 'no_config'}
        from solders.keypair import Keypair
        from solders.pubkey import Pubkey
        from solders.instruction import Instruction, AccountMeta
        from solders.transaction import Transaction
        from solders.message import Message
        from solana.rpc.api import Client
        import hashlib
        client = Client(SOLANA_RPC)
        oracle = Keypair.from_bytes(oracle_keypair)
        program_id = Pubkey.from_string(PROGRAM_ID)
        state_pubkey = Pubkey.from_string(STATE_ACCOUNT)
        machine_pubkey_obj = Pubkey.from_string(machine_pubkey)
        job_pubkey_obj = Pubkey.from_string(job_pubkey)
        machine_state_pda, _ = Pubkey.find_program_address(
            [b"machine", bytes(machine_pubkey_obj)], program_id
        )
        discriminator = hashlib.sha256(b"global:update_trust").digest()[:8]
        job_hash_bytes = job_hash.encode('utf-8')
        instruction_data = (
            discriminator +
            struct.pack('<I', len(job_hash_bytes)) + job_hash_bytes +
            struct.pack('<I', ml_confidence) + struct.pack('<b', trust_delta)
        )
        accounts = [
            AccountMeta(state_pubkey, is_signer=False, is_writable=False),
            AccountMeta(machine_state_pda, is_signer=False, is_writable=True),
            AccountMeta(job_pubkey_obj, is_signer=False, is_writable=True),
            AccountMeta(oracle.pubkey(), is_signer=True, is_writable=False),
        ]
        instruction = Instruction(program_id, instruction_data, accounts)
        blockhash_resp = client.get_latest_blockhash()
        message = Message.new_with_blockhash([instruction], oracle.pubkey(), blockhash_resp.value.blockhash)
        tx = Transaction.new_unsigned(message)
        tx.sign([oracle], blockhash_resp.value.blockhash)
        result = client.send_transaction(tx)
        return {'status': 'sent', 'signature': str(result.value)}
    except Exception as e:
        print(f'[DEBUG] update_trust error: {e}')
        if 'AccountNotInitialized' in str(e):
            return {'status': 'skipped', 'reason': 'machine_not_registered'}
        return {'status': 'error', 'reason': str(e)[:200]}

def call_settle_job(machine_pubkey, job_pubkey, owner_pubkey):
    """Call settle_job to distribute MINT tokens"""
    try:
        if not oracle_keypair or not STATE_ACCOUNT:
            return {'status': 'skipped', 'reason': 'no_config'}
        
        from solders.keypair import Keypair
        from solders.pubkey import Pubkey
        from solders.instruction import Instruction, AccountMeta
        from solders.transaction import Transaction
        from solders.message import Message
        from solana.rpc.api import Client
        import hashlib
        
        client = Client(SOLANA_RPC)
        oracle = Keypair.from_bytes(oracle_keypair)
        program_id = Pubkey.from_string(PROGRAM_ID)
        state_pubkey = Pubkey.from_string(STATE_ACCOUNT)
        machine_pubkey_obj = Pubkey.from_string(machine_pubkey)
        job_pubkey_obj = Pubkey.from_string(job_pubkey)
        mint_pubkey = Pubkey.from_string(MINT_TOKEN)
        genesis_treasury_ata = Pubkey.from_string(GENESIS_TREASURY_ATA)
        personal_fee_ata = Pubkey.from_string(PERSONAL_FEE_ATA)
        protocol_fee_ata = Pubkey.from_string(PROTOCOL_FEE_ATA)
        token_program = Pubkey.from_string(TOKEN_PROGRAM_ID)
        
        # Derive PDAs
        machine_state_pda, _ = Pubkey.find_program_address(
            [b"machine", bytes(machine_pubkey_obj)], program_id
        )
        genesis_authority_pda, _ = Pubkey.find_program_address(
            [b"genesis_authority"], program_id
        )
        mint_authority_pda, _ = Pubkey.find_program_address(
            [b"mint_authority"], program_id
        )
        
        # Get owner's MINT token account (ATA)
        owner_pubkey_obj = Pubkey.from_string(owner_pubkey)
        owner_token_account = get_associated_token_address(owner_pubkey_obj, mint_pubkey)
        
        # Build instruction
        discriminator = hashlib.sha256(b"global:settle_job").digest()[:8]
        
        # SettleJob accounts (must match Rust struct order):
        # 1. state (mut)
        # 2. machine_state (mut, PDA)
        # 3. job (mut)
        # 4. mint (mut)
        # 5. genesis_treasury_ata (mut)
        # 6. genesis_authority (PDA)
        # 7. owner_token_account (mut)
        # 8. personal_fee_token_account (mut)
        # 9. protocol_fee_token_account (mut)
        # 10. mint_authority (PDA)
        # 11. token_program
        
        accounts = [
            AccountMeta(state_pubkey, is_signer=False, is_writable=True),
            AccountMeta(machine_state_pda, is_signer=False, is_writable=True),
            AccountMeta(job_pubkey_obj, is_signer=False, is_writable=True),
            AccountMeta(mint_pubkey, is_signer=False, is_writable=True),
            AccountMeta(genesis_treasury_ata, is_signer=False, is_writable=True),
            AccountMeta(genesis_authority_pda, is_signer=False, is_writable=False),
            AccountMeta(owner_token_account, is_signer=False, is_writable=True),
            AccountMeta(personal_fee_ata, is_signer=False, is_writable=True),
            AccountMeta(protocol_fee_ata, is_signer=False, is_writable=True),
            AccountMeta(mint_authority_pda, is_signer=False, is_writable=False),
            AccountMeta(token_program, is_signer=False, is_writable=False),
        ]
        
        instruction = Instruction(program_id, discriminator, accounts)
        blockhash_resp = client.get_latest_blockhash()
        message = Message.new_with_blockhash([instruction], oracle.pubkey(), blockhash_resp.value.blockhash)
        tx = Transaction.new_unsigned(message)
        tx.sign([oracle], blockhash_resp.value.blockhash)
        result = client.send_transaction(tx)
        
        return {'status': 'sent', 'signature': str(result.value)}
    except Exception as e:
        return {'status': 'error', 'reason': str(e)[:200]}

def extract_job_data(tx):
    from solders.pubkey import Pubkey
    instructions = tx.get('instructions', [])
    for ix in instructions:
        if ix.get('programId', '') == PROGRAM_ID:
            accounts = ix.get('accounts', [])
            if len(accounts) >= 5:
                decoded = decode_record_job_instruction(ix.get('data', ''))
                if decoded:
                    job_hash = decoded.get('job_hash')
                    program_id = Pubkey.from_string(PROGRAM_ID)
                    job_pda, _ = Pubkey.find_program_address(
                        [b"job", job_hash.encode()], program_id
                    )
                    print(f"[DEBUG] job_hash: {job_hash}, derived job_pda: {job_pda}")
                    return {
                        'machine_id': str(accounts[3]),
                        'owner_id': str(accounts[4]),
                        'job_hash': job_hash,
                        'job_pubkey': str(job_pda),
                        'duration_seconds': decoded.get('duration_seconds', 300),
                        'complexity_claimed': decoded.get('complexity_claimed', 1.0),
                    }
    return None

# ============ API Routes ============

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'version': config.get('version'),
        'model': 'v4',
        'economic_model': 'v2_time_anchor',
        'base_rate': BASE_RATE,
        'machines': len(machine_history),
        'jobs': network_stats['total_jobs'],
        'total_mint_minted': round(network_stats['total_mint_minted'], 2),
        'disputes': len(community_disputes),
        'oracle_configured': oracle_keypair is not None,
        'state_configured': bool(STATE_ACCOUNT),
        'settle_enabled': True
    })

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        payload = request.json
        if not payload:
            return jsonify({'status': 'no payload'}), 200
        transactions = payload if isinstance(payload, list) else [payload]
        results = []
        for tx in transactions:
            signature = tx.get('signature', 'unknown')
            job_data = extract_job_data(tx)
            if job_data:
                result = score_job(job_data)
                result['job_hash'] = job_data.get('job_hash')
                result['machine_id'] = job_data.get('machine_id')
                result['owner_id'] = job_data.get('owner_id')
                result['job_pubkey'] = job_data.get('job_pubkey')
                result['signature'] = signature
                result['duration_seconds'] = job_data.get('duration_seconds')
                result['complexity_claimed'] = job_data.get('complexity_claimed')
                scored_jobs.append({'timestamp': datetime.now().isoformat(), **result})
                if len(scored_jobs) > 1000:
                    scored_jobs.pop(0)
                
                # Step 1: Update trust score on-chain
                # Wait for job to be confirmed on-chain
                print("[DEBUG] Waiting 3s for job confirmation...")
                time.sleep(3)
                trust_result = call_update_trust(
                    job_data.get('machine_id'),
                    job_data.get('job_pubkey'),
                    job_data.get('job_hash'),
                    int(result['confidence'] * 1000),
                    result['trust_delta']
                )
                result['trust_update'] = trust_result
                print(f"[DEBUG] trust_result: {trust_result}")
                
                # Step 2: Settle job and distribute MINT (if trust update succeeded)
                if trust_result.get('status') == 'sent':
                    # Small delay to let trust update confirm
                    time.sleep(5)
                    settle_result = call_settle_job(
                        job_data.get('machine_id'),
                        job_data.get('job_pubkey'),
                        job_data.get('owner_id')
                    )
                    result['settle'] = settle_result
                    print(f"[DEBUG] settle_result: {settle_result}")
                    if settle_result.get('status') == 'sent':
                        print(f"[SETTLE] TX: {settle_result.get('signature', 'unknown')[:16]}...")
                else:
                    result['settle'] = {'status': 'skipped', 'reason': 'trust_update_failed'}
                
                econ = result['economics']
                print(f"[MINT] #{econ['network_jobs']} | {job_data.get('machine_id','')[:8]}... | "
                      f"{econ['duration_seconds']}s @ {econ['complexity_claimed']:.2f} | "
                      f"norm:{econ['normalized_complexity']:.2f} | "
                      f"base:{econ['base_reward']:.4f} | trust:{econ['trust_score']} | "
                      f"final:{econ['final_reward']:.4f} MINT | "
                      f"ML:{result['confidence']*100:.1f}%→{result['action']}")
                results.append(result)
        return jsonify({'status': 'processed', 'results': results, 'count': len(results)})
    except Exception as e:
        print(f"WEBHOOK ERROR: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def stats():
    net_avg = get_network_avg_complexity()
    return jsonify({
        'network_avg_complexity': round(net_avg, 3),
        'network_avg_duration': round(np.mean(network_stats['durations'][-1000:]) if network_stats['durations'] else 300, 1),
        'machines': len(machine_history),
        'total_jobs': network_stats['total_jobs'],
        'total_duration_hours': round(network_stats['total_duration'] / 3600, 2),
        'total_mint_minted': round(network_stats['total_mint_minted'], 2),
        'base_rate': BASE_RATE,
        'disputes': len(community_disputes)
    })

@app.route('/recent-scores')
def recent_scores():
    return jsonify({'count': len(scored_jobs), 'jobs': scored_jobs[-50:]})

@app.route('/economy')
def economy():
    net_avg = get_network_avg_complexity()
    machines_summary = []
    for mid, h in machine_history.items():
        machines_summary.append({
            'id': mid[:12] + '...',
            'jobs': h['job_count'],
            'trust': h['trust_score'],
            'warmup': round(calculate_warmup(h['job_count']), 2),
            'total_earned': round(h['total_earned'], 4),
            'total_duration_hours': round(h['total_duration'] / 3600, 2),
            'avg_complexity': round(np.mean(h['complexities'][-20:]) if h['complexities'] else 1.0, 2),
            'avg_duration': round(np.mean(h['durations'][-20:]) if h['durations'] else 300, 0),
            'earning_rate_per_hour': round(h['total_earned'] / (h['total_duration'] / 3600) if h['total_duration'] > 0 else 0, 4)
        })
    machines_summary.sort(key=lambda x: -x['total_earned'])
    return jsonify({
        'constants': {
            'base_rate': BASE_RATE,
            'base_rate_per_hour': BASE_RATE * 3600,
            'warmup_jobs': WARMUP_JOBS,
            'decay_halflife_days': DECAY_HALFLIFE_DAYS,
            'complexity_bounds': [MIN_COMPLEXITY_MULT, MAX_COMPLEXITY_MULT]
        },
        'network': {
            'total_jobs': network_stats['total_jobs'],
            'total_duration_hours': round(network_stats['total_duration'] / 3600, 2),
            'total_mint_minted': round(network_stats['total_mint_minted'], 4),
            'total_machines': len(machine_history),
            'avg_complexity_7d': round(net_avg, 3),
            'avg_duration': round(np.mean(network_stats['durations'][-1000:]) if network_stats['durations'] else 300, 1),
        },
        'machines': machines_summary[:20],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/machine/<machine_id>')
def machine_info(machine_id):
    h = machine_history.get(machine_id)
    if not h:
        return jsonify({'error': 'Machine not found'}), 404
    return jsonify({
        'machine_id': machine_id,
        'job_count': h['job_count'],
        'trust_score': h['trust_score'],
        'warmup': round(calculate_warmup(h['job_count']), 3),
        'total_earned': round(h['total_earned'], 6),
        'total_duration_hours': round(h['total_duration'] / 3600, 2),
        'avg_complexity': round(np.mean(h['complexities'][-20:]) if h['complexities'] else 1.0, 3),
        'avg_duration': round(np.mean(h['durations'][-20:]) if h['durations'] else 300, 1),
        'earning_rate_per_hour': round(h['total_earned'] / (h['total_duration'] / 3600) if h['total_duration'] > 0 else 0, 6)
    })

@app.route('/calculate-reward', methods=['POST'])
def calculate_reward():
    try:
        data = request.json
        duration = float(data.get('duration_seconds', 300))
        complexity = float(data.get('complexity_claimed', 1.0))
        trust = int(data.get('trust_score', 100))
        job_count = int(data.get('job_count', 0))
        age_days = float(data.get('age_days', 0))
        base_result = calculate_base_reward(duration, complexity, job_count)
        final_result = calculate_final_reward(base_result['base_reward'], trust, age_days)
        return jsonify({**base_result, **final_result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/dispute', methods=['POST'])
def submit_dispute():
    try:
        data = request.json
        if not data or not data.get('job_hash') or not data.get('dispute_type'):
            return jsonify({'error': 'job_hash and dispute_type required'}), 400
        dispute = {
            'timestamp': datetime.now().isoformat(),
            'job_hash': data['job_hash'],
            'dispute_type': data['dispute_type'],
            'reason': data.get('reason', ''),
        }
        community_disputes.append(dispute)
        if len(community_disputes) > 1000:
            community_disputes.pop(0)
        return jsonify({'status': 'recorded', 'total_disputes': len(community_disputes)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/disputes')
def get_disputes():
    return jsonify({'count': len(community_disputes), 'disputes': community_disputes[-100:]})

@app.route('/export-training-data')
def export_training_data():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        'timestamp', 'job_hash', 'machine_id', 'duration_seconds', 'complexity_claimed',
        'normalized_complexity', 'base_reward', 'final_reward', 'trust_score',
        'ml_confidence', 'ml_action', 'trust_delta', 'disputed', 'dispute_type'
    ])
    dispute_lookup = {d['job_hash']: d for d in community_disputes}
    for job in scored_jobs:
        d = dispute_lookup.get(job.get('job_hash', ''), {})
        econ = job.get('economics', {})
        writer.writerow([
            job.get('timestamp', ''), job.get('job_hash', ''), job.get('machine_id', ''),
            job.get('duration_seconds', ''), job.get('complexity_claimed', ''),
            econ.get('normalized_complexity', ''), econ.get('base_reward', ''),
            econ.get('final_reward', ''), econ.get('trust_score', ''),
            job.get('confidence', ''), job.get('action', ''), job.get('trust_delta', ''),
            'yes' if d else 'no', d.get('dispute_type', '')
        ])
    output.seek(0)
    return Response(output.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': f'attachment; filename=training_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
