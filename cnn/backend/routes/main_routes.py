from flask import Blueprint, jsonify, Response
from services.monitoring import get_history, get_last_activity

main_bp = Blueprint('main', __name__)

@main_bp.route("/status", methods=["GET"])
def get_status():
    """Get model and system status"""
    history = get_history()
    
    return jsonify({
        "total_recognized": len(history),
        "last_recognition": history[-1] if history else None
    })

@main_bp.route("/history", methods=["GET"])
def get_history_endpoint():
    return jsonify({"history": get_history()})

@main_bp.route("/last_activity", methods=["GET"])
def get_last_activity_endpoint():
    activity, last_update_value = get_last_activity()
    
    return jsonify({
    "latest_update": last_update_value,
    "recent_history": activity})

@main_bp.route("/metrics", methods=["GET"])
def get_metrics():
    """Get simple Prometheus metrics"""
    try:
        history = get_history()
        
        if history and len(history) > 0:
            last = history[-1]
            if isinstance(last, dict) and 'digits' in last:
                # [1,5,5,1,9] -> 15519
                value = int(''.join(map(str, last['digits'])))
            else:
                value = last
        else:
            value = 0

        metrics = f"""# HELP last_recognition_value Last recognized value
# TYPE last_recognition_value gauge
last_recognition_value {value}
"""
        return Response(metrics, mimetype='text/plain')
        
    except Exception as e:
        return Response("# Error\n", mimetype='text/plain', status=500)