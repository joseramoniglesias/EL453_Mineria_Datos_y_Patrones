import psycopg2
from config import config
 
def conectar():
    """ Conexión al servidor de pases de datos PostgreSQL """
    conexion = None
    try:
        # Lectura de los parámetros de conexion
        params = config()
 
        # Conexion al servidor de PostgreSQL
        print('Conectando a la base de datos PostgreSQL...')
        conexion = psycopg2.connect(host="localhost",database="empleados", user="postgres", password="1234")
 
        # Crea del cursor
        cur = conexion.cursor()

        # Ejecuta una consulta
        print('Los datos de la BD son:')
        cur.execute( 'SELECT codigo,nombre FROM personal' )

        # Muestra los resultados
        for codigo, nombre in cur.fetchall() :
            print (codigo, nombre)

        # Cierra la conexión
        conexion.close()
        # Cierre de la comunicación con PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conexion is not None:
            conexion.close()
            print('Conexión finalizada.')

if __name__ == 'consultaBD':
    conectar()