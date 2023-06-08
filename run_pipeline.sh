#!/bin/sh

echo "Enter filename with fluorescence calcium transients (without '.npy'): "
echo "[Make sure each row corresponds to one neuron]"

# read in data 
read filename
echo "File name is ${filename}.npy"
traces = python3 -c'import ./src/dataload.py; dataload.load_data('${filename}.npy')'


echo "Enter sampling frequency (in Hz):"
read f_s
echo "Frequency = $f_s Hz."

echo "remove motion artifact ..."
read decision
if decision is True:
do 
    python3 -c'import ./src/dataPreProcessing.py; dataPreProcessing.replace_nan_(traces,bad_frames,delete_frames=True)'
else:
do





matlab -nodisplay -nodesktop -r "try remove_motion_artifact('$filename'); catch; end; quit"

echo "z-score fluorescence ..."

matlab -nodisplay -nodesktop -r "try zscore_f('$filename'); catch; end; quit"

echo "smooth the signal with total-variation regularization ..."

matlab -nodesktop -r "try tvreg_smoothen('$filename', $sf); catch; end; quit"

echo "Now you see the noise correlation, is it large enough that you want to use the smoothened fluorescence, instead of the original fluorescence? (y/n)"

read yn

case $yn in
    [yY] ) echo "replace signal with the smooth version ...";
	   filename="${filename}_smooth";;
    [nN] ) echo "keep the original signal, use the z-scored version...";
	   filename="${filename}_zs";;
    * ) echo "invalid response";
esac

echo "Filename is ${filename}.txt. This is to be typed in the next user input."
	   
echo "Now we compute the granger causality..."
#echo "enter lags: "
#read lags
#echo "lags = ${lags}"
#echo "enter p-value threshold:"
#read pthres
#echo "p-value threshold = ${pthres}"

./compute_gc.py