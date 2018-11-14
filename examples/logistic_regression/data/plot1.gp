

# refer to
# https://stackoverflow.com/questions/9082807/gnuplot-plotting-points-with-color-based-values-in-one-column

w01 = 0.176031
w02 = -0.0095
w11 = 0.19624
w12 = 0.018

w00 = 1.414
w10 = 0.020

f(x) =- ((w11 - w01)*x + (w10 - w00)) / (w12 - w02)

#set xrange [30:100]
#set yrange [30:100]
set datafile separator ","
plot "< awk -F ',' '{if($3 == \"1\") print}' ex2data1.txt" u 1:2 t "Admitted" w p pt 7 ps 1, \
     "< awk -F ',' '{if($3 == \"0\") print}' ex2data1.txt" u 1:2 t "Not admitted" w p pt 5 ps 1, f(x);
