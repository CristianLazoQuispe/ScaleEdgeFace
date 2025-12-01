import imageio
import os

def convert_avi_to_mp4(input_path, output_path,fps=20):
    # Crear un objeto writer con formato MP4
    print("convert avi to mp4 : fps = ",fps/4)
    writer = imageio.get_writer(output_path, fps=fps/4)

    # Leer los frames del video AVI y escribirlos en el nuevo archivo MP4
    with imageio.get_reader(input_path) as reader:
        for frame in reader:
            writer.append_data(frame)
    print("saved video :",output_path)

    # Cerrar el objeto writer
    writer.close()

# Rutas para el archivo de entrada AVI y el archivo de salida MP4
input_avi_path = "results/output_time/jetson_b01/face_detection_analisis/video_original_normal/total.avi"
output_mp4_path = input_avi_path.replace(".avi",".mp4")

# Convertir AVI a MP4
convert_avi_to_mp4(input_avi_path, output_mp4_path,6)