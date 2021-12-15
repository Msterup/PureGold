import schedule
import redis
import time


r = redis.Redis(host='127.0.0.1', port=6379, db=0, password='MikkelSterup')

def doTraining(state):
    if state == "run":
        r.set('is_run', 1)
    else:
        r.set('is_run', 0)



schedule.every().day.at("07:30:00").do(doTraining, "run")
schedule.every().day.at("18:30:00").do(doTraining, "not_run")

while True:
    schedule.run_pending()
    time.sleep(1)