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

app = Flask(__name__)
CORS(app)

model = joblib.load('model.pkl')
with open('model_metadata.json', 'r') as f:
    config = json.load(f)
with open('model_features.json', 'r') as f:
    feature_list = json.load(f)

print(f"Model loaded. {len(feature_list)} features.")

PROGRAM_ID = os.environ.get('PROGRAM_ID', 'AyFBC6DBStSbrau3wfFZzsX5rX14nx8Gkp8TqF687F5X')
STATE_ACCOUNT = os.environ.get('STATE_ACCOUNT', '')
SOLANA_RPC = os.environ.get('SOLANA_RPC', 'https://api.devnet.solana.com')
ORACLE_PRIVATE_KEY = os.environ.get('ORACLE_PRIVATE_KEY', '')

oracle_keypair = None
if ORACLE_PRIVATE_KEY:
    try:
        key_bytes = json.loads(ORACLE_PRIVATE_KEY)
        oracle_keypair = bytes(key_bytes)
        print(f"Oracle keypair loaded: {len(oracle_keypair)} bytes")
    except Exception as e:
        print(f"Failed to load oracle keypair: {e}")

# Economic constants
ACTIVITY_WINDOW_SECONDS = 7 * 24 * 60 * 60  # 7 days
DECAY_HALFLIFE_DAYS = 5
BASELINE_JOBS = 100  # For activity ratio

machine_history = defaultdict(lambda: {
    'job_count': 0, 'total_earned': 0, 
    'complexities': [], 'durations': [],
    'timestamps': [], 'trust_score': 100
})
network_stats = {
    'complexities': [], 'durations': [], 
    'timestamps': [], 'total_jobs': 0
}
scored_jobs = []
community_disputes = []
economic_snapshots = []

def calculate_activity_ratio():
    """Activity ratio: min(1.0, BASELINE / jobs_in_window)"""
    now = time.time()
    window_start = now - ACTIVITY_WINDOW_SECONDS
    jobs_in_window = sum(1 for t in network_stats['timestamps'] if t > window_start)
    if jobs_in_window == 0:
        return 1.0
    ratio = min(1.0, BASELINE_JOBS / jobs_in_window)
    return round(ratio, 4)

def calculate_decay(job_age_seconds):
    """Exponential decay with 5-day half-life"""
    days_old = job_age_seconds / (24 * 60 * 60)
    decay = np.exp(-0.693 * days_old / DECAY_HALFLIFE_DAYS)
    return round(decay, 4)

def calculate_warmup(job_count):
    """Warmup: 0.5 → 1.0 over 30 jobs"""
    warmup = 0.5 + 0.5 * min(1.0, job_count / 30)
    return round(warmup, 4)

def get_network_stats():
    if not network_stats['complexities']:
        return {'avg_c': 1.0, 'std_c': 0.3, 'avg_d': 500, 'std_d': 300}
    c = network_stats['complexities'][-1000:]
    d = network_stats['durations'][-1000:]
    return {
        'avg_c': np.mean(c), 'std_c': np.std(c) or 0.3,
        'avg_d': np.mean(d), 'std_d': np.std(d) or 300
    }

def get_machine_stats(mid):
    h = machine_history[mid]
    net = get_network_stats()
    if h['job_count'] < 10:
        return net
    c, d = h['complexities'][-100:], h['durations'][-100:]
    return {
        'avg_c': np.mean(c), 'std_c': np.std(c) or net['std_c'],
        'avg_d': np.mean(d), 'std_d': np.std(d) or net['std_d']
    }

def build_features(data):
    mid = data.get('machine_id', 'unknown')
    c = float(data.get('complexity_claimed', 1.0))
    dur = float(data.get('duration_seconds', 300))
    rew = float(data.get('reward_gross', 5.0))
    
    net, mach = get_network_stats(), get_machine_stats(mid)
    h = machine_history[mid]
    
    # Calculate economic multipliers
    activity_ratio = calculate_activity_ratio()
    warmup = calculate_warmup(h['job_count'])
    
    return {
        'complexity_claimed': c, 'duration_seconds': dur, 'duration_minutes': dur/60,
        'reward_gross': rew, 'reward_per_second': rew/(dur+1), 'reward_per_complexity': rew/(c+0.1),
        'network_avg_complexity': net['avg_c'], 'network_avg_duration': net['avg_d'],
        'network_complexity_std': net['std_c'], 'network_duration_std': net['std_d'],
        'jobs_in_window': data.get('jobs_in_window', 100),
        'days_since_launch': data.get('days_since_launch', 1.0),
        'activity_ratio': activity_ratio,
        'decay_multiplier': data.get('decay_multiplier', 1.0),
        'warmup_multiplier': warmup,
        'machine_job_count': h['job_count'], 'machine_total_earned': h['total_earned'],
        'machine_avg_complexity': mach['avg_c'], 'machine_avg_duration': mach['avg_d'],
        'machine_complexity_std': mach['std_c'], 'machine_duration_std': mach['std_d'],
        'complexity_vs_network': c - net['avg_c'], 'complexity_vs_machine': c - mach['avg_c'],
        'complexity_zscore_network': (c - net['avg_c']) / (net['std_c'] + 0.01),
        'complexity_zscore_machine': (c - mach['avg_c']) / (mach['std_c'] + 0.01),
        'duration_vs_network': dur - net['avg_d'], 'duration_vs_machine': dur - mach['avg_d'],
        'duration_zscore_network': (dur - net['avg_d']) / (net['std_d'] + 1),
        'duration_zscore_machine': (dur - mach['avg_d']) / (mach['std_d'] + 1),
        'time_since_last_job': data.get('time_since_last_job', 0),
        'jobs_last_hour_machine': data.get('jobs_last_hour_machine', 0),
        'is_new_machine': 1 if h['job_count'] < 10 else 0,
        'complexity_duration_ratio': c / (dur/60 + 0.1),
        'reward_efficiency': rew / (dur + 1),
        'earning_rate': h['total_earned'] / (h['job_count'] + 1),
    }

def update_history(mid, c, dur, rew):
    now = time.time()
    h = machine_history[mid]
    h['job_count'] += 1
    h['total_earned'] += rew
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
    if len(network_stats['complexities']) > 5000:
        network_stats['complexities'] = network_stats['complexities'][-5000:]
        network_stats['durations'] = network_stats['durations'][-5000:]
        network_stats['timestamps'] = network_stats['timestamps'][-5000:]

def score_job(data):
    mid = data.get('machine_id', 'unknown')
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
    
    # Update machine trust tracking
    h = machine_history[mid]
    old_trust = h['trust_score']
    h['trust_score'] = max(0, min(100, old_trust + trust_delta))
    
    update_history(mid, 
        float(data.get('complexity_claimed', 1.0)),
        float(data.get('duration_seconds', 300)),
        float(data.get('reward_gross', 5.0)))
    
    # Calculate economic snapshot
    activity_ratio = calculate_activity_ratio()
    warmup = calculate_warmup(h['job_count'])
    net = get_network_stats()
    
    return {
        'confidence': confidence,
        'trust_delta': trust_delta,
        'action': action,
        'economics': {
            'activity_ratio': activity_ratio,
            'warmup_multiplier': warmup,
            'machine_trust': h['trust_score'],
            'machine_jobs': h['job_count'],
            'network_jobs': network_stats['total_jobs'],
            'network_avg_complexity': round(net['avg_c'], 3),
            'network_avg_duration': round(net['avg_d'], 1),
        }
    }

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
    except Exception as e:
        return None

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
            struct.pack('<I', ml_confidence) + struct.pack('<i', trust_delta)
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
        if 'AccountNotInitialized' in str(e):
            return {'status': 'skipped', 'reason': 'machine_not_registered'}
        return {'status': 'error', 'reason': str(e)[:100]}

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'version': config.get('version'),
        'machines': len(machine_history),
        'jobs': network_stats['total_jobs'],
        'disputes': len(community_disputes),
        'oracle_configured': oracle_keypair is not None,
        'state_configured': bool(STATE_ACCOUNT)
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
                result['job_pubkey'] = job_data.get('job_pubkey')
                result['signature'] = signature
                result['duration_seconds'] = job_data.get('duration_seconds')
                result['complexity_claimed'] = job_data.get('complexity_claimed')
                
                scored_jobs.append({
                    'timestamp': datetime.now().isoformat(),
                    **result
                })
                if len(scored_jobs) > 1000:
                    scored_jobs.pop(0)
                
                trust_result = call_update_trust(
                    job_data.get('machine_id'),
                    job_data.get('job_pubkey'),
                    job_data.get('job_hash'),
                    int(result['confidence'] * 1000),
                    result['trust_delta']
                )
                result['trust_update'] = trust_result
                
                # VERBOSE ECONOMIC LOG
                econ = result['economics']
                print(f"[ECON] Job #{econ['network_jobs']} | "
                      f"Machine: {job_data.get('machine_id','')[:8]}... | "
                      f"Jobs: {econ['machine_jobs']} | "
                      f"Trust: {econ['machine_trust']} | "
                      f"Activity: {econ['activity_ratio']:.3f} | "
                      f"Warmup: {econ['warmup_multiplier']:.2f} | "
                      f"Net Avg C: {econ['network_avg_complexity']:.2f} | "
                      f"ML: {result['confidence']*100:.1f}% → {result['action']}")
                
                results.append(result)
        
        return jsonify({'status': 'processed', 'results': results, 'count': len(results)})
    except Exception as e:
        print(f"WEBHOOK ERROR: {e}")
        return jsonify({'error': str(e)}), 500

def extract_job_data(tx):
    instructions = tx.get('instructions', [])
    for ix in instructions:
        if ix.get('programId', '') == PROGRAM_ID:
            accounts = ix.get('accounts', [])
            if len(accounts) >= 4:
                decoded = decode_record_job_instruction(ix.get('data', ''))
                if decoded:
                    return {
                        'machine_id': str(accounts[3]),
                        'job_hash': decoded.get('job_hash'),
                        'job_pubkey': str(accounts[2]),
                        'duration_seconds': decoded.get('duration_seconds', 300),
                        'complexity_claimed': decoded.get('complexity_claimed', 1.0),
                        'reward_gross': 5.0,
                        'activity_ratio': 1.0,
                        'decay_multiplier': 1.0
                    }
    return None

@app.route('/stats')
def stats():
    net = get_network_stats()
    return jsonify({
        'network_avg_complexity': round(net['avg_c'], 3),
        'network_avg_duration': round(net['avg_d'], 1),
        'network_std_complexity': round(net['std_c'], 3),
        'network_std_duration': round(net['std_d'], 1),
        'machines': len(machine_history),
        'total_jobs': network_stats['total_jobs'],
        'activity_ratio': calculate_activity_ratio(),
        'disputes': len(community_disputes)
    })

@app.route('/recent-scores')
def recent_scores():
    return jsonify({'count': len(scored_jobs), 'jobs': scored_jobs[-50:]})

@app.route('/economy')
def economy():
    """Verbose economic state endpoint"""
    net = get_network_stats()
    activity = calculate_activity_ratio()
    
    # Machine summaries
    machines_summary = []
    for mid, h in machine_history.items():
        machines_summary.append({
            'id': mid[:12] + '...',
            'jobs': h['job_count'],
            'trust': h['trust_score'],
            'warmup': calculate_warmup(h['job_count']),
            'avg_complexity': round(np.mean(h['complexities'][-20:]) if h['complexities'] else 1.0, 2),
            'avg_duration': round(np.mean(h['durations'][-20:]) if h['durations'] else 300, 0),
        })
    
    machines_summary.sort(key=lambda x: -x['jobs'])
    
    return jsonify({
        'network': {
            'total_jobs': network_stats['total_jobs'],
            'total_machines': len(machine_history),
            'activity_ratio': activity,
            'avg_complexity': round(net['avg_c'], 3),
            'avg_duration': round(net['avg_d'], 1),
            'std_complexity': round(net['std_c'], 3),
            'std_duration': round(net['std_d'], 1),
        },
        'machines': machines_summary[:20],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/machine/<machine_id>')
def machine_info(machine_id):
    h = machine_history.get(machine_id, {})
    if not h:
        return jsonify({'error': 'Machine not found'}), 404
    return jsonify({
        'machine_id': machine_id,
        'job_count': h.get('job_count', 0),
        'trust_score': h.get('trust_score', 100),
        'warmup': calculate_warmup(h.get('job_count', 0)),
        'total_earned': h.get('total_earned', 0),
        'avg_complexity': round(np.mean(h.get('complexities', [1.0])[-20:]), 3),
        'avg_duration': round(np.mean(h.get('durations', [300])[-20:]), 1),
    })

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
    writer.writerow(['timestamp', 'job_hash', 'machine_id', 'duration_seconds', 'complexity_claimed',
                     'ml_confidence', 'ml_action', 'trust_delta', 'disputed', 'dispute_type'])
    
    dispute_lookup = {d['job_hash']: d for d in community_disputes}
    for job in scored_jobs:
        d = dispute_lookup.get(job.get('job_hash', ''), {})
        writer.writerow([
            job.get('timestamp', ''), job.get('job_hash', ''), job.get('machine_id', ''),
            job.get('duration_seconds', ''), job.get('complexity_claimed', ''),
            job.get('confidence', ''), job.get('action', ''), job.get('trust_delta', ''),
            'yes' if d else 'no', d.get('dispute_type', '')
        ])
    
    output.seek(0)
    return Response(output.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': f'attachment; filename=training_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
