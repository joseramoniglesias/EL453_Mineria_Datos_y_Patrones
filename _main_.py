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
 
        # creación del cursor
        cur = conexion.cursor()
        
        # Ejecución de una consulta con la version de PostgreSQL
        print('La version de PostgreSQL es la:')
        cur.execute('SELECT version()')
 
        # Ahora mostramos la version
        version = cur.fetchone()
        print(version)
       
        # Cierre de la comunicación con PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conexion is not None:
            conexion.close()
            print('Conexión finalizada.')
 
 
if __name__ == '__main__':
    conectar()
