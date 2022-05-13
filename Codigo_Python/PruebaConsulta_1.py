import psycopg2

conexion = psycopg2.connect(host="localhost",database="empleados", user="postgres", password="1234")
# creación del cursor
cur = conexion.cursor()
cur.execute("select * from personal")
#cur.execute("delete from personal where codigo=1")
#cur.execute("update personal set nombre='Juan Pablo' where codigo=2")
cur.execute("insert into personal (nombre) values('Johnatan Torres')")
cur.execute("insert into personal (nombre) values('Jairo Quintero')")
cur.execute("insert into personal (nombre) values('Andres Garcia')")
conexion.commit()
cur.execute("select codigo, nombre from personal")
for fila in cur:
    print(fila)
conexion.close()    


