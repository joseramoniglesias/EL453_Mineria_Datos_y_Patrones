from configparser import ConfigParser
 
def config(archivo='base_de_datos.ini', seccion='postgresql'):
    # Crear el parser y leer el archivo
    parser = ConfigParser()
    parser.read(archivo)
# Obtener la sección de conexión a la base de datos
    db = {}
    if parser.has_section(seccion):
        params = parser.items(seccion)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Secccion {0} no encontrada en el archivo {1}'.format(seccion, archivo))
