import time


class AlertSystem:
    """
    Alert notification system for fall and abnormal movement events
    """
    
    def __init__(self):
        self.last_alert_time = {}
        self.alert_history = []
    
    def should_alert(self, alert_type, cooldown_seconds=2.5):
        """
        Check if alert should be triggered (respects cooldown)
        """
        current_time = time.time()
        last_time = self.last_alert_time.get(alert_type, 0)
        
        if current_time - last_time > cooldown_seconds:
            self.last_alert_time[alert_type] = current_time
            return True
        return False
    
    def log_alert(self, status, risk_score):
        """
        Log alert event to console and history
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        alert_msg = f"[{timestamp}] ALERT: {status} | Risk: {risk_score:.2f}"
        print(f"🚨 {alert_msg}")
        self.alert_history.append(alert_msg)
