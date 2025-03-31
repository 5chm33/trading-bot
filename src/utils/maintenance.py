# src/utils/maintenance.py
import schedule
import time
from utils.logging import setup_logger

logger = setup_logger(__name__)

def nightly_routine():
    """11 PM system maintenance"""
    try:
        logger.info("Running nightly maintenance")
        
        # 1. Validate today's trades
        from pipeline.evaluation import validate_trades
        errors = validate_trades()
        
        # 2. Backup database
        from utils.storage import backup_db
        backup_db()
        
        # 3. Generate report
        from reporting.daily import generate_report
        generate_report()
        
        if errors:
            from alerts import send_alert
            send_alert(f"Nightly errors detected: {len(errors)}")
            
    except Exception as e:
        logger.critical(f"Nightly routine failed: {e}")

# Schedule daily at 11 PM
schedule.every().day.at("23:00").do(nightly_routine)

if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(60)