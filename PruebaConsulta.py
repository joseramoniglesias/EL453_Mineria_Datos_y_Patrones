import psycopg2
# Creamos la conexión
conexion = psycopg2.connect(host="localhost",database="empleados", user="postgres", password="1234")
# creación del cursor
cur = conexion.cursor()
# Ejecutamos una consulta y guaramos los resultados
cur.execute("SELECT codigo, nombre FROM personal")

for fila in cur:
    print(fila)
# Cerramos la conexión
conexion.close()

