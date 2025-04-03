import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
from torchvision.transforms import functional as F
import pandas as pd
import glob

import matplotlib
matplotlib.use('Agg')

## image connection
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.discovery import build
from google.oauth2 import service_account
import numpy as np
from sqlalchemy import create_engine, text
from dash.exceptions import PreventUpdate


# Conexión a la base de datos
engine = create_engine('postgresql://wikijs:wikijsrocks@34.28.115.42:5432/wiki')


# Coordenadas
def faster_rcnn_to_yolo(bbox_rcnn, img_width, img_height):
    """
    Convierte de formato Faster R-CNN (x_min, y_min, x_max, y_max)
    a formato YOLO (x_center, y_center, width, height).
    
    bbox_rcnn: [x_min, y_min, x_max, y_max] en formato Faster R-CNN.
    img_width: Ancho de la imagen.
    img_height: Alto de la imagen.
    
    Retorna: [x_center, y_center, width, height] en formato YOLO.
    """
    x_min, y_min, x_max, y_max = bbox_rcnn
    
    # Calcular las dimensiones absolutas
    width_abs = x_max - x_min
    height_abs = y_max - y_min
    
    # Calcular el centro
    x_center_abs = x_min + width_abs / 2
    y_center_abs = y_min + height_abs / 2
    
    # Convertir a coordenadas relativas (dividiendo por el ancho y alto de la imagen)
    x_center = x_center_abs / img_width
    y_center = y_center_abs / img_height
    width = width_abs / img_width
    height = height_abs / img_height
    
    return [x_center, y_center, width, height]


# Etiquetas
def get_common_name(category : str):
    dict_common_names = {'Blattodea': 'Cucarachas (y similares)',
                         'Coleoptera': 'Cucarrones, mariquitas (escarabajos en general)',
                         'Dermaptera' : 'Tijeretas',
                         'Diglossa' : 'Pajaritos "picaflor"',
                         'Diptera' : 'Moscas, mosquitos, zancudos (y similares)',
                         'Hemiptera': 'Chinches, cigarras',
                         'Hymenoptera' : 'Abejas, abejorros, avispas',
                         'Lepidoptera': 'Mariposas, polillas',
                         'Mantodea': 'Mantis',
                         'Megaloptera': 'Moscas de Dobson ?',
                         'Neuroptera' : 'Crisopas ?',
                         'Odonata' : 'Libélulas, caballitos del diablo',
                         'Orthoptera' : 'Grillos, saltamontes (y similares)',
                         'Phasmida': 'Insectos palo',
                         'Phyllostomidae' : 'Murciélagos',
                         'Trochilidae' : 'Colibríes'}
    
    return dict_common_names[category]

# Base de datos
def get_image_data():
    
    query = 'SELECT * FROM polinizadores.validacion_imagenes WHERE validated_box = 0 AND discarded_image = 0 AND discarded_box = 0 AND is_in_use = 0 ORDER BY RANDOM() LIMIT 1'
    
    df = pd.read_sql(query, engine)
    list_info = ({'id': df.id.iloc[0], 'image_path' : df.image_path.iloc[0], 'url': df['url_image'].iloc[0].split('/')[-2], 
                   'bbox': (df.xmin_r.iloc[0], df.ymin_r.iloc[0], df.xmax_r.iloc[0], df.ymax_r.iloc[0]),
                   'category': df['category'].iloc[0]})
    
    image_id = list_info['id']
    image_path = list_info['image_path']

    with engine.begin() as connection:
        query = text(f"""
                UPDATE polinizadores.validacion_imagenes 
                        SET is_in_use = 1 
                        WHERE id = :image_id AND image_path = :image_path
            """)
        connection.execute(query, {'image_id':int(image_id), 'image_path':image_path})

    return list_info

# Conexión Drive para imágenes
creds = service_account.Credentials.from_service_account_file(
    'conexion_drive/polinizadores-438316-8355b2af125c.json',
    scopes=['https://www.googleapis.com/auth/drive']
)

drive_service = build('drive', 'v3', credentials=creds)

def retrieve_image_from_drive(file_id: str):
    image_data = io.BytesIO()
    request = drive_service.files().get_media(fileId=file_id)
    downloader = MediaIoBaseDownload(image_data, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%.")

    image_data.seek(0)
    image = Image.open(image_data)
    return image

# Función para cargar la imagen
def load_image(file_id):
    image = retrieve_image_from_drive(file_id)
    image = image.convert("RGB")
    width, height = image.size
    image_tensor = F.to_tensor(image).unsqueeze(0)  # Convertir a tensor y añadir dimensión batch
    image_np = image_tensor.squeeze().permute(1, 2, 0).numpy()
    return image_np, width, height

# Crear la app Dash
app = dash.Dash(__name__)

list_info = get_image_data()
print('al llamar get_image_data en 126 devuelve', list_info['image_path'])

app.layout = html.Div([
    # Instrucciones para el usuario
    html.Div([
        html.H2("Instrucciones:"),
        html.P('1. Usa el botón "Caja Correcta" si la caja cubre los bordes del insecto en cuestión. Usa "Caja Incorrecta" si la caja no cubre los bordes del insecto.'),
        html.P('2. Usa "Imagen NO Válida" si la imagen es de un insecto que no corresponde a la categoría, es una imagen de microscopio o no es insecto en "estado natural".'),
        html.P('3. Si no aparecen cajas en la imagen, dibuja las necesarias que encierren los insectos visibles que corresponden a la categoría. Luego, haz clic en "Guardar y Siguiente Imagen".')
    ], style={'fontSize': '15px', 'margin-bottom': '5px', 'border': '2px solid #ccc', 'padding': '2px', 'border-radius': '2px', 'backgroundColor': '#f9f9f9'}),
    
     html.H1(id='image-title', style={'fontSize': '22px'}),
    
    dcc.Graph(id="image-graph"),  # Gráfico interactivo
    
    html.Button('Caja Correcta', id='correct-button', n_clicks=0, style={'fontSize': 15, 'margin-right': '50px', 'backgroundColor': 'green', 'color': 'white'}),
    html.Button('Caja Incorrecta', id='incorrect-button', n_clicks=0, style={'fontSize': 15, 'margin-right': '50px', 'backgroundColor': 'red', 'color': 'white'}),
    html.Button('Imagen NO Válida', id='validate-image-button', n_clicks=0, style={'fontSize': 15, 'margin-right': '50px', 'backgroundColor': 'orange', 'color': 'white'}),
    html.Button('No Estoy Segurx, Siguiente Imagen', id='no-sure-button', n_clicks=0, style={'fontSize': 15, 'margin-right': '50px', 'backgroundColor': 'gray', 'color': 'white'}), 
    html.Button('Guardar y Siguiente Imagen', id='next-button', n_clicks=0, style={'display': 'none', 'fontSize': 20, 'margin-right': '50px', 'backgroundColor': 'blue', 'color': 'white'}), 
     
    
    html.Div(id='validation-status'),  # Estado de validación
    html.Div(id='output-coordinates'),  # Coordenadas del rectángulo dibujado

    # Añadir dcc.Store para manejar el trigger de validación
    dcc.Store(id='validation-trigger', data={'trigger': True})
])

# Callback para actualizar el título
@app.callback(
    Output('image-title', 'children'),
    [Input('validation-trigger', 'data')]  # Disparador cuando se actualizan los datos
)
def update_title(trigger_data):
    if trigger_data['trigger']:
        category = list_info['category']
        common_name = get_common_name(category)
        image_id = list_info['id']
        image_path = list_info['image_path']
        return f"Validación de Bounding Boxes - {category} ({common_name}) ID : {image_id}, Path : {image_path}"
    raise PreventUpdate


# Callback para mostrar la imagen actual con posibilidad de dibujar cajas si las coordenadas son nulas
@app.callback(
    [Output('image-graph', 'figure'), Output('next-button', 'style')],
    [Input('next-button', 'n_clicks'), Input('validation-trigger', 'data')]  # Añadimos el trigger como Input
)
def display_image(n_clicks, trigger_data):
    
    global list_info

    print('display muestra', list_info['id'], list_info['image_path'])
    
    # Cargar la imagen y las coordenadas
    if trigger_data['trigger'] or n_clicks > 0:
        print('entro al display')
        image_np, width, height = load_image(list_info['url'])
        list_info['width'] = width
        list_info['height'] = height
        
        # Usamos Plotly Express para mostrar la imagen
        fig = px.imshow(image_np)
        
        # Obtener las coordenadas de la imagen actual
        coords = list_info['bbox']
        print(coords[0])
        # Si no hay bounding boxes (coordenadas nulas), habilitar el dibujo
        if pd.isna(coords[0]):
            # Permitir que el usuario dibuje rectángulos
            fig.update_layout(
                dragmode='drawrect',
                newshape=dict(line_color='cyan')
            )
            # Mostrar el botón "Guardar y Siguiente Imagen"
            button_style = {'display': 'inline-block'}
        else:
            # Si ya tiene una bounding box, la dibujamos
            xmin, ymin, xmax, ymax = coords
            fig.add_shape(
                type="rect",
                x0=xmin, y0=ymin, x1=xmax, y1=ymax,
                line=dict(color="red", width=2)
            )
            # Ocultar el botón "Guardar y Siguiente Imagen"
            button_style = {'display': 'none'}
        
        return fig, button_style
    
    raise PreventUpdate

# Callback para capturar las coordenadas del rectángulo dibujado
@app.callback(
    Output('output-coordinates', 'children'),
    [Input('image-graph', 'relayoutData')]
)
def capture_rectangle(relayoutData):
    if relayoutData and 'shapes' in relayoutData:
        # Extraemos las coordenadas del último rectángulo dibujado
        shape = relayoutData['shapes'][-1]
        xmin = shape['x0']
        ymin = shape['y0']
        xmax = shape['x1']
        ymax = shape['y1']
        
        # Aquí puedes agregar lógica para guardar estas coordenadas en la base de datos
        return f"Rectángulo dibujado: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}"
    
    #return "Dibuja un rectángulo en la imagen."



@app.callback(
    [Output('validation-status', 'children'), Output('validation-trigger', 'data')],  # Añadimos esta salida
    [Input('correct-button', 'n_clicks'), Input('incorrect-button', 'n_clicks'),
     Input('validate-image-button', 'n_clicks'), Input('no-sure-button', 'n_clicks')],
    [State('output-coordinates', 'children')]
)
def validate_and_next_image(correct_clicks, incorrect_clicks, validate_image_clicks, no_sure_clicks, coordinates):
    global list_info
    trigger_next_image = {'trigger': False}  # Inicializamos el trigger

    if len(list_info) == 0:
        raise Exception("No hay más imágenes para validar")  
    
    image_id = list_info['id']
    image_path = list_info['image_path']

    # Identificamos qué botón fue presionado usando callback_context
    ctx = dash.callback_context

    if not ctx.triggered:
        raise PreventUpdate  # Si no hubo interacción con ningún botón, no actualizamos nada

    # Obtenemos el id del botón que disparó el callback
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Si se hizo clic en "Caja Correcta"
    if button_id == 'correct-button':
        print('entra a correct_clicks')
        with engine.begin() as connection:
            query = text(f"""
                UPDATE polinizadores.validacion_imagenes 
                SET validated_box = 1, is_in_use = 0
                WHERE id = :image_id AND image_path = :image_path
            """)
            connection.execute(query, {'image_id': int(image_id), 'image_path': image_path})

        msg = f"Caja de la imagen {image_id} ({image_path}) marcada como Correcta."
        print(msg)

        # Pasar a la siguiente imagen
        list_info = get_image_data()
        trigger_next_image = {'trigger': True}
        return msg, trigger_next_image

    # Si se hizo clic en "Caja Incorrecta"
    elif button_id == 'incorrect-button':
        print('entra a incorrect_clicks')
        with engine.begin() as connection:
            query = text(f"""
                UPDATE polinizadores.validacion_imagenes 
                SET discarded_box = 1, is_in_use = 0 
                WHERE id = :image_id AND image_path = :image_path
            """)
            connection.execute(query, {'image_id': int(image_id), 'image_path': image_path})

        msg = f"Caja de la imagen {image_id} ({image_path}) marcada como Incorrecta."
        print(msg)

        # Pasar a la siguiente imagen
        list_info = get_image_data()
        trigger_next_image = {'trigger': True}
        return msg, trigger_next_image

    # Si se hizo clic en "Validar Imagen" como inválida
    elif button_id == 'validate-image-button':
        print('entra a validar imagen')
        with engine.begin() as connection:
            query = text(f"""
                UPDATE polinizadores.validacion_imagenes 
                SET discarded_image = 1, validated_box = 1, is_in_use = 0 
                WHERE image_path = :image_path
            """)
            connection.execute(query, {'image_path': image_path})

        msg = f"Imagen {image_id} ({image_path}) marcada como Inválida."
        print(msg)

        # Pasar a la siguiente imagen
        list_info = get_image_data()
        trigger_next_image = {'trigger': True}
        return msg, trigger_next_image

    # Si se hizo clic en "No Estoy Segurx"
    elif button_id == 'no-sure-button':
        print('entra a cambiar imagen por not sure')
        with engine.begin() as connection:
            query = text(f"""
                UPDATE polinizadores.validacion_imagenes 
                SET is_in_use = 0 
                WHERE id = :image_id AND image_path = :image_path
            """)
            connection.execute(query, {'image_id': int(image_id), 'image_path': image_path})

        msg = 'Cambiando Imagen'
        print(msg)

        # Pasar a la siguiente imagen
        list_info = get_image_data()
        trigger_next_image = {'trigger': True}
        return msg, trigger_next_image
     
    else:
        print('entra al else con', image_id, image_path)
        image_id = list_info['id']
        image_path = list_info['image_path']

        trigger_next_image = {'trigger': True}
        print('cambiar el trigger y sale del else con', image_id, image_path)
        
        return '', trigger_next_image

# Callback para manejar el botón de "Guardar y Siguiente Imagen" cuando se dibuja una caja
@app.callback(
    Output('next-button', 'n_clicks'),
    [Input('next-button', 'n_clicks')],
    [State('output-coordinates', 'children')]
)
def save_and_next_image(next_clicks, coordinates):
    global list_info

    image_id =  list_info['id']
    image_path = list_info['image_path']

    # Lógica para guardar las coordenadas en la base de datos
    if coordinates:
        # Extraer coordenadas
        coords = coordinates.replace("Rectángulo dibujado: ", "").split(", ")
        coords = [float(c.split("=")[1]) for c in coords]
        xmin, ymin, xmax, ymax = coords

        image_id = int(image_id)  # Asegúrate de que sea un tipo int nativo de Python
        xmin = float(xmin)
        ymin = float(ymin)
        xmax = float(xmax)
        ymax = float(ymax)
    

        yolo_coords = faster_rcnn_to_yolo([ xmin, ymin, xmax, ymax], list_info['width'], list_info['height'])
        yolo_coords = [float(coord) for coord in yolo_coords] 

        # Guardar las coordenadas en la base de datos
        with engine.begin() as connection:
            query = text(f"""
                UPDATE polinizadores.validacion_imagenes 
                SET xmin_r = :xmin, ymin_r = :ymin, xmax_r = :xmax, ymax_r = :ymax ,
                         xcenter_y = :xcenter_y , ycenter_y = :ycenter_y, width_y = :width_y, height_y = :height_y,
                         validated_box = 1, is_in_use = 0
                WHERE id = :image_id AND image_path = :image_path
            """)
            connection.execute(query, {
            'xmin': xmin, 
            'ymin': ymin, 
            'xmax': xmax, 
            'ymax': ymax, 
            'xcenter_y': yolo_coords[0], 
            'ycenter_y': yolo_coords[1], 
            'width_y': yolo_coords[2], 
            'height_y': yolo_coords[3],
            'image_id': image_id, 
            'image_path': image_path
            })
                               
        print(f"Guardando coordenadas: {coordinates} para {image_id} {image_path}")

        # Pasar a la siguiente imagen
        print('está ejecutando esto que está dentro de dibujar coordenadas, entra con', image_id, image_path)
        list_info = get_image_data()
        image_id = list_info['id']
        image_path = list_info['image_path']


    if len(list_info) == 0:
        raise Exception("No hay más imágenes para validar")   
    return next_clicks  # Mantener el conteo de clicks para activar el callback display_image

if __name__ == '__main__':
    app.run_server(debug=True)
