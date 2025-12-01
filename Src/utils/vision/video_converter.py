import imageio
import os
def convert_avi_to_mp4(input_path, output_path,fps=20):
    # Crear un objeto writer con formato MP4
    print("convert avi to mp4 : fps = ",fps/3.3)
    writer = imageio.get_writer(output_path, fps=fps/3.3)

    # Leer los frames del video AVI y escribirlos en el nuevo archivo MP4
    with imageio.get_reader(input_path) as reader:
        for frame in reader:
            writer.append_data(frame)
    print("saved video :",output_path)

    # Cerrar el objeto writer
    writer.close()