import os
import gdown

# Lista de enlaces de Google Drive para los videos
video_links = [
    'https://drive.google.com/file/d/1O80LjYGKySattIDgE2Mrf57ugQGRotB5/view?usp=sharing',
    'https://drive.google.com/file/d/1LA9oQSIkRYF6pP-x-O1RI1UyTyyB0g48/view?usp=drive_link',
    'https://drive.google.com/file/d/1tQRTGoHkcjVXNdBRF0-LJBvDTm9EkI9T/view?usp=drive_link',
    'https://drive.google.com/file/d/1OW3yByy0muFDrFcgOHBEqID5aRqmnAWv/view?usp=drive_link',
    'https://drive.google.com/file/d/1uaj8ZRbE-533EVt2VTMQYmMf72pfA9J1/view?usp=drive_link'
]

# Carpeta local donde se guardarán los videos
output_folder = 'videos/'
os.makedirs(output_folder, exist_ok=True)

# Descargar cada video
for video_link in video_links:
    video_id = video_link.split('/')[-2]
    gdown.download(f'https://drive.google.com/uc?id={video_id}', output_folder, quiet=False)

print("Descarga completada.")
