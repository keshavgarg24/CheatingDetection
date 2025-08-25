import sqlite3, pprint
con = sqlite3.connect('monitoring.sqlite')
print("\nSTATUS:")
pprint.pp(con.execute('select student_id,strikes,status,last_update from monitoring_status').fetchall())
print("\nVIOLATIONS:")
pprint.pp(con.execute('select student_id,violation_type,detail,ts from violations order by id desc limit 20').fetchall())
con.close()
