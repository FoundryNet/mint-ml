from flask import Flask, request, jsonify
import joblib
import json
import numpy as np
from datetime import datetime
from collections import defaultdict
import base64

app = Flask(__name__)

model = joblib.load('model.pkl')
with open('model_metadata.json', 'r') as f:
    config = json.load(f)
with open('model_features.json', 'r') as f:
    feature_list = json.load(f)

print(f"Model loaded. {len(feature_list)} features.")

machine_history = defaultdict(lambda: {
    'job_count': 0, 'total_earned': 0, 
    'complexities': [], 'durations': []
})
network_stats = {'complexities': [], 'durations': []}

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
    """Score a job and return ML results"""
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

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'version': config.get('version'),
        'machines': len(machine_history), 
        'jobs': len(network_stats['complexities'])
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Direct prediction endpoint"""
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
        print(f"Webhook received: {json.dumps(payload)[:500]}")
        
        if not payload:
            return jsonify({'status': 'no payload'}), 200
        
        # Helius sends array of transactions
        transactions = payload if isinstance(payload, list) else [payload]
        
        results = []
        for tx in transactions:
            # Extract from Helius enhanced format
            # Look for our program's events in the transaction
            
            # Check if this is our program
            account_keys = tx.get('accountKeys', [])
            program_id = 'AyFBC6DBStSbrau3wfFZzsX5rX14nx8Gkp8TqF687F5X'
            
            if program_id not in str(account_keys):
                continue
            
            # Try to extract JobRecorded event data from logs
            logs = tx.get('meta', {}).get('logMessages', []) or tx.get('logMessages', [])
            
            # Parse event data from logs
            job_data = None
            for log in logs:
                if 'JobRecorded' in log or 'job_hash' in log:
                    # Found our event
                    job_data = parse_job_from_logs(logs, tx)
                    break
            
            if not job_data:
                # Try to extract from transaction instructions
                job_data = parse_job_from_tx(tx)
            
            if job_data:
                result = score_job(job_data)
                result['job_hash'] = job_data.get('job_hash')
                result['machine_id'] = job_data.get('machine_id')
                result['signature'] = tx.get('signature')
                results.append(result)
                
                print(f"Scored job: {result}")
        
        return jsonify({
            'status': 'processed',
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        print(f"Webhook error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def parse_job_from_logs(logs, tx):
    """Extract job data from transaction logs"""
    try:
        # Anchor emits events as base64 in logs
        for log in logs:
            if 'Program data:' in log:
                # This is event data
                data_part = log.split('Program data: ')[-1]
                # Decode and parse (simplified - may need adjustment)
                pass
        
        # Fallback: extract from instruction data
        return parse_job_from_tx(tx)
    except:
        return None

def parse_job_from_tx(tx):
    """Extract job data from transaction structure"""
    try:
        # Get instructions
        instructions = tx.get('instructions', [])
        
        for ix in instructions:
            # Look for record_job instruction
            data = ix.get('data', '')
            accounts = ix.get('accounts', [])
            
            if len(accounts) >= 4:  # record_job has state, machine_state, job, machine, payer
                # Extract machine pubkey
                machine_id = accounts[3] if len(accounts) > 3 else 'unknown'
                
                # For now, use placeholder values - we'll refine this
                return {
                    'machine_id': str(machine_id),
                    'job_hash': tx.get('signature', 'unknown')[:32],
                    'duration_seconds': 300,  # Will extract from ix data
                    'complexity_claimed': 1.0,
                    'reward_gross': 5.0,
                    'activity_ratio': 1.0,
                    'decay_multiplier': 1.0
                }
        
        return None
    except Exception as e:
        print(f"Parse error: {e}")
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

# Test endpoint to simulate webhook
@app.route('/test-webhook', methods=['POST'])
def test_webhook():
    """Test endpoint to simulate a job"""
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
