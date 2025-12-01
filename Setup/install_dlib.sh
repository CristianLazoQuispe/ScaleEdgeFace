sudo apt-get update
sudo apt-get install -y cmake libavdevice-dev libavfilter-dev libavformat-dev libavcodec-dev libswresample-dev libswscale-dev libavutil-dev
sudo apt-get install -y build-essential cmake pkg-config libx11-dev libatlas-base-dev libgtk-3-dev libboost-python-dev

git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build
cd build
cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
cmake --build . --config Release
make 
sudo make install
sudo ldconfig
cd ..
#python setup.py install --yes USE_AVX_INSTRUCTIONS --yes DLIB_USE_CUDA
python setup.py install --set DLIB_USE_CUDA=1
