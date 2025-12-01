sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'
sudo nvpmodel -m 0
sudo jetson_clocks
