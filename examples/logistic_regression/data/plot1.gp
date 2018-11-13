

# refer to
# https://stackoverflow.com/questions/9082807/gnuplot-plotting-points-with-color-based-values-in-one-column

set datafile separator ","
plot "< awk -F ',' '{if($3 == \"1\") print}' ex2data1.txt" u 1:2 t "Admitted" w p pt 7 ps 1, \
     "< awk -F ',' '{if($3 == \"0\") print}' ex2data1.txt" u 1:2 t "Not admitted" w p pt 5 ps 1;
