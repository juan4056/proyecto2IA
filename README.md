# PROYECTO 2

## Comandos:

```bash
# Execute 10 experiments with Decision Tree
g++ decisiontree_script.cpp -o decisiontree.o
./decisiontree.o -test 0
./decisiontree.o -test 1
./decisiontree.o -test 2
./decisiontree.o -test 3
./decisiontree.o -test 4
./decisiontree.o -test 5
./decisiontree.o -test 6
./decisiontree.o -test 7
./decisiontree.o -test 8
./decisiontree.o -test 9
# Execute 10 experiments with KNN
g++ -c -I./ann_1.1.2/include -O3 knn_script.cpp
g++ knn_script.o -o knn_script -L./ann_1.1.2/lib -lANN -lm
./knn_script -test 0
./knn_script -test 1
./knn_script -test 2
./knn_script -test 3
./knn_script -test 4
./knn_script -test 5
./knn_script -test 6
./knn_script -test 7
./knn_script -test 8
./knn_script -test 9

# Install ANN
cd ann_1.1.2 && make
```
