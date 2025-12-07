from flask import Flask, request, jsonify
import joblib
import json
import numpy as np
from datetime import datetime
from collections import defaultdict
import base58
import struct
import os

app = Flask(__name__)

model = joblib.load('model.pkl')
with open('model_metadata.json', 'r') as f:
    config = json.load(f)
with open('model_features.json', 'r') as f:
    feature_list = json.load(f)

print(f"Model loaded. {len(feature_list)} features.")

# Program ID
PROGRAM_ID = 'AyFBC6DBStSbrau3wfFZzsX5rX14nx8Gkp8TqF687F5X'

# Oracle keypair for signing update_trust transactions
# In production, load from environment variable
ORACLE_PRIVATE_KEY = os.environ.get('ORACLE_PRIVATE_KEY', None)

machine_history = defaultdict(lambda: {
    'job_count': 0, 'total_earned': 0, 
    'complexities': [], 'durations': []
})
network_stats = {'complexities': [], 'durations': []}
scored_jobs = []  # Store recent scores for inspection

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
    
    return {
        'complexity_claimed': c, 'duration_seconds': dur, 'duration_minutes': dur/60,
        'reward_gross': rew, 'reward_per_second': rew/(dur+1), 'reward_per_complexity': rew/(c+0.1),
        'network_avg_complexity': net['avg_c'], 'network_avg_duration': net['avg_d'],
        'network_complexity_std': net['std_c'], 'network_duration_std': net['std_d'],
        'jobs_in_window': data.get('jobs_in_window', 100),
        'days_since_launch': data.get('days_since_launch', 1.0),
        'activity_ratio': data.get('activity_ratio', 1.0),
        'decay_multiplier': data.get('decay_multiplier', 1.0),
        'warmup_multiplier': data.get('warmup_multiplier', 1.0),
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
    h = machine_history[mid]
    h['job_count'] += 1
    h['total_earned'] += rew
    h['complexities'].append(c)
    h['durations'].append(dur)
    if len(h['complexities']) > 500:
        h['complexities'] = h['complexities'][-500:]
        h['durations'] = h['durations'][-500:]
    network_stats['complexities'].append(c)
    network_stats['durations'].append(dur)
    if len(network_stats['complexities']) > 5000:
        network_stats['complexities'] = network_stats['complexities'][-5000:]
        network_stats['durations'] = network_stats['durations'][-5000:]

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
    
    update_history(mid, 
        float(data.get('complexity_claimed', 1.0)),
        float(data.get('duration_seconds', 300)),
        float(data.get('reward_gross', 5.0)))
    
    return {
        'confidence': confidence,
        'trust_delta': trust_delta,
        'action': action
    }

def decode_record_job_instruction(data_b58):
    """Decode record_job instruction data from base58"""
    try:
        data = base58.b58decode(data_b58)
        print(f"Decoded bytes length: {len(data)}")
        print(f"Decoded bytes hex: {data.hex()}")
        
        # Anchor discriminator is first 8 bytes
        discriminator = data[:8]
        print(f"Discriminator: {discriminator.hex()}")
        
        # After discriminator comes the arguments
        # record_job(job_hash: String, duration_sec: u64, complexity_fixed: u32)
        
        offset = 8
        
        # String is: 4 bytes length (u32 LE) + bytes
        if len(data) > offset + 4:
            str_len = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            job_hash = data[offset:offset+str_len].decode('utf-8', errors='ignore')
            offset += str_len
            print(f"Job hash: {job_hash}")
            
            # duration_sec: u64 (8 bytes LE)
            if len(data) >= offset + 8:
                duration_sec = struct.unpack('<Q', data[offset:offset+8])[0]
                offset += 8
                print(f"Duration: {duration_sec}")
                
                # complexity_fixed: u32 (4 bytes LE)
                if len(data) >= offset + 4:
                    complexity_fixed = struct.unpack('<I', data[offset:offset+4])[0]
                    print(f"Complexity fixed: {complexity_fixed}")
                    
                    return {
                        'job_hash': job_hash,
                        'duration_seconds': duration_sec,
                        'complexity_fixed': complexity_fixed,
                        'complexity_claimed': complexity_fixed / 1000.0  # Convert from fixed
                    }
        
        return None
    except Exception as e:
        print(f"Decode error: {e}")
        return None

def call_update_trust(machine_pubkey, job_hash, ml_confidence, trust_delta):
    """Call update_trust on Solana (async, fire-and-forget for now)"""
    try:
        if not ORACLE_PRIVATE_KEY:
            print("No oracle key configured - skipping on-chain update")
            return {'status': 'skipped', 'reason': 'no_oracle_key'}
        
        # TODO: Implement actual Solana transaction
        # For now, just log what we would do
        print(f"Would call update_trust:")
        print(f"  Machine: {machine_pubkey}")
        print(f"  Job: {job_hash}")
        print(f"  ML Confidence: {ml_confidence}")
        print(f"  Trust Delta: {trust_delta}")
        
        return {'status': 'pending', 'reason': 'not_implemented_yet'}
        
    except Exception as e:
        print(f"update_trust error: {e}")
        return {'status': 'error', 'reason': str(e)}

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'version': config.get('version'),
        'machines': len(machine_history), 
        'jobs': len(network_stats['complexities']),
        'oracle_configured': ORACLE_PRIVATE_KEY is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data'}), 400
        
        result = score_job(data)
        result['job_hash'] = data.get('job_hash')
        result['machine_id'] = data.get('machine_id', 'unknown')
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/webhook', methods=['POST'])
def webhook():
    """Helius webhook endpoint - receives JobRecorded events"""
    try:
        payload = request.json
        
        print("=" * 60)
        print("WEBHOOK RECEIVED")
        print("=" * 60)
        
        if not payload:
            return jsonify({'status': 'no payload'}), 200
        
        transactions = payload if isinstance(payload, list) else [payload]
        
        results = []
        for tx in transactions:
            signature = tx.get('signature', 'unknown')
            print(f"Processing tx: {signature[:20]}...")
            
            # Extract job data
            job_data = extract_job_data(tx)
            
            if job_data:
                print(f"Job data: {job_data}")
                result = score_job(job_data)
                result['job_hash'] = job_data.get('job_hash')
                result['machine_id'] = job_data.get('machine_id')
                result['signature'] = signature
                result['duration_seconds'] = job_data.get('duration_seconds')
                result['complexity_claimed'] = job_data.get('complexity_claimed')
                
                # Store for inspection
                scored_jobs.append({
                    'timestamp': datetime.now().isoformat(),
                    **result
                })
                if len(scored_jobs) > 100:
                    scored_jobs.pop(0)
                
                # Call update_trust on-chain
                trust_result = call_update_trust(
                    job_data.get('machine_id'),
                    job_data.get('job_hash'),
                    int(result['confidence'] * 1000),
                    result['trust_delta']
                )
                result['trust_update'] = trust_result
                
                results.append(result)
                print(f"SCORED: {result}")
        
        return jsonify({
            'status': 'processed',
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        print(f"WEBHOOK ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def extract_job_data(tx):
    """Extract job data from Helius transaction"""
    
    instructions = tx.get('instructions', [])
    
    for ix in instructions:
        program_id = ix.get('programId', '')
        if program_id == PROGRAM_ID:
            accounts = ix.get('accounts', [])
            data_b58 = ix.get('data', '')
            
            # accounts for record_job: [state, machine_state, job, machine, payer, system]
            if len(accounts) >= 4:
                machine_pubkey = accounts[3] if len(accounts) > 3 else None
                job_pubkey = accounts[2] if len(accounts) > 2 else None
                
                # Decode instruction data
                decoded = decode_record_job_instruction(data_b58)
                
                if decoded:
                    return {
                        'machine_id': str(machine_pubkey),
                        'job_hash': decoded.get('job_hash', str(job_pubkey)),
                        'job_pubkey': str(job_pubkey),
                        'duration_seconds': decoded.get('duration_seconds', 300),
                        'complexity_claimed': decoded.get('complexity_claimed', 1.0),
                        'reward_gross': 5.0,  # Will calculate from duration/complexity
                        'activity_ratio': 1.0,
                        'decay_multiplier': 1.0
                    }
                else:
                    # Fallback to defaults if decode fails
                    return {
                        'machine_id': str(machine_pubkey),
                        'job_hash': str(job_pubkey),
                        'duration_seconds': 300,
                        'complexity_claimed': 1.0,
                        'reward_gross': 5.0,
                        'activity_ratio': 1.0,
                        'decay_multiplier': 1.0
                    }
    
    return None

@app.route('/stats')
def stats():
    net = get_network_stats()
    return jsonify({
        'network_avg_complexity': net['avg_c'],
        'network_avg_duration': net['avg_d'],
        'machines': len(machine_history),
        'jobs': len(network_stats['complexities'])
    })

@app.route('/recent-scores')
def recent_scores():
    """View recently scored jobs"""
    return jsonify({
        'count': len(scored_jobs),
        'jobs': scored_jobs[-20:]  # Last 20
    })

@app.route('/machine/<machine_id>')
def machine_info(machine_id):
    """Get machine history and stats"""
    h = machine_history.get(machine_id, {})
    if not h:
        return jsonify({'error': 'Machine not found'}), 404
    
    return jsonify({
        'machine_id': machine_id,
        'job_count': h.get('job_count', 0),
        'total_earned': h.get('total_earned', 0),
        'avg_complexity': np.mean(h.get('complexities', [1.0])) if h.get('complexities') else 1.0,
        'avg_duration': np.mean(h.get('durations', [300])) if h.get('durations') else 300,
        'recent_complexities': h.get('complexities', [])[-10:],
        'recent_durations': h.get('durations', [])[-10:]
    })

@app.route('/test-webhook', methods=['POST'])
def test_webhook():
    data = request.json or {
        'machine_id': 'test-machine-001',
        'job_hash': 'test-job-' + str(datetime.now().timestamp()),
        'duration_seconds': 300,
        'complexity_claimed': 1.2,
        'reward_gross': 5.0
    }
    
    result = score_job(data)
    result['job_hash'] = data.get('job_hash')
    result['machine_id'] = data.get('machine_id')
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
