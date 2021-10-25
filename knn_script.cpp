#include <cstdlib>   // C standard library
#include <cstdio>    // C I/O (for sscanf)
#include <cstring>   // string manipulation
#include <fstream>   // file I/O
#include <ANN/ANN.h> // ANN declarations

using namespace std; // make std:: accessible

void getArgs(int argc, char **argv); // get command-line arguments

int k = 3;         // number of nearest neighbors
int dim = 7;       // dimension
double eps = 0;    // error bound
int maxPts = 1000; // maximum number of data points
int classes = 2;

istream *dataIn = NULL;   // input for data points
istream *dataVals = NULL; // input for data values
istream *queryIn = NULL;  // input for query points
ostream *answers = NULL;  // output for query answers

bool readPt(istream &in, ANNpoint p) // read point (false on EOF)
{
    for (int i = 0; i < dim; i++)
    {
        if (!(in >> p[i]))
            return false;
    }
    return true;
}

bool readValue(istream &in, int &value)
{
    if (!(in >> value))
        return false;
    return true;
}

void printPt(ostream &out, ANNpoint p) // print point
{
    out << "(" << p[0];
    for (int i = 1; i < dim; i++)
    {
        out << ", " << p[i];
    }
    out << ")\n";
}

bool writeValue(ostream &out, int &value)
{
    if (!(out << value))
        return false;
    if (!(out << "\n"))
        return false;
    return true;
}

int main(int argc, char **argv)
{
    int nPts;              // actual number of data points
    ANNpointArray dataPts; // data points
    ANNpoint queryPt;      // query point
    ANNidxArray nnIdx;     // near neighbor indices
    ANNdistArray dists;    // near neighbor distances
    ANNkd_tree *kdTree;    // search structure
    int *classification_vals;
    int *values_knn;

    getArgs(argc, argv); // read command-line arguments

    queryPt = annAllocPt(dim);             // allocate query point
    dataPts = annAllocPts(maxPts, dim);    // allocate data points
    nnIdx = new ANNidx[k];                 // allocate near neigh indices
    dists = new ANNdist[k];                // allocate near neighbor dists
    classification_vals = new int[maxPts]; // allocate values classification from data points (male=0, female=1)

    nPts = 0; // read data points

    cout << "Data Points:\n";
    while (nPts < maxPts && readPt(*dataIn, dataPts[nPts]) && readValue(*dataVals, classification_vals[nPts]))
    {
        printPt(cout, dataPts[nPts]);
        nPts++;
    }

    kdTree = new ANNkd_tree( // build search structure
        dataPts,             // the data points
        nPts,                // number of points
        dim);                // dimension of space

    while (readPt(*queryIn, queryPt))
    {                            // read query points
        cout << "Query point: "; // echo query point
        printPt(cout, queryPt);

        kdTree->annkSearch( // search
            queryPt,        // query point
            k,              // number of near neighbors
            nnIdx,          // nearest neighbors (returned)
            dists,          // distance (returned)
            eps);           // error bound

        cout << "\tNN:\tIndex\tDistance\tValue\n";
        values_knn = new int[classes];
        for (int i = 0; i < k; i++)
        {                              // print summary
            dists[i] = sqrt(dists[i]); // unsquare distance
            cout << "\t" << i << "\t" << nnIdx[i] << "\t" << dists[i];
            cout << "\t" << classification_vals[nnIdx[i]] << "\n";
            values_knn[classification_vals[nnIdx[i]]]++;
        }
        int max = -1, value = -1;
        for (int i = 0; i < classes; i++)
        {
            if (values_knn[i] > max)
            {
                max = values_knn[i];
                value = i;
            }
        }
        cout << "\tResult:\t" << value << "\n";
        writeValue(*answers, value);
    }
    delete[] nnIdx; // clean things up
    delete[] dists;
    delete kdTree;
    annClose(); // done with ANN

    return EXIT_SUCCESS;
}

//----------------------------------------------------------------------
//	getArgs - get command line arguments
//----------------------------------------------------------------------

void getArgs(int argc, char **argv)
{
    static ifstream dataStream;    // data file stream
    static ifstream dataVStream;   // data file stream
    static ifstream queryStream;   // query file stream
    static ofstream answersStream; // answers file stream

    if (argc <= 1)
    { // no arguments
        cerr << "Usage:\n\n"
             << "  ann_sample [-d dim] [-max m] [-nn k] [-e eps] [-df data]"
                " [-qf query]\n\n"
             << "  where:\n"
             << "    dim      dimension of the space (default = 2)\n"
             << "    m        maximum number of data points (default = 1000)\n"
             << "    k        number of nearest neighbors per query (default 1)\n"
             << "    eps      the error bound (default = 0.0)\n"
             << "    data     name of file containing data points\n"
             << "    query    name of file containing query points\n\n"
             << " Results are sent to the standard output.\n"
             << "\n"
             << " To run this demo use:\n"
             << "    ann_sample -df data.pts -qf query.pts\n";
        exit(0);
    }
    int i = 1;
    while (i < argc)
    { // read arguments
        if (!strcmp(argv[i], "-d"))
        {                          // -d option
            dim = atoi(argv[++i]); // get dimension to dump
        }
        else if (!strcmp(argv[i], "-max"))
        {                             // -max option
            maxPts = atoi(argv[++i]); // get max number of points
        }
        else if (!strcmp(argv[i], "-nn"))
        {                        // -nn option
            k = atoi(argv[++i]); // get number of near neighbors
        }
        else if (!strcmp(argv[i], "-e"))
        {                                   // -e option
            sscanf(argv[++i], "%lf", &eps); // get error bound
        }
        else if (!strcmp(argv[i], "-df"))
        {                                        // -df option
            dataStream.open(argv[++i], ios::in); // open data file
            if (!dataStream)
            {
                cerr << "Cannot open data file\n";
                exit(1);
            }
            dataIn = &dataStream; // make this the data stream
        }
        else if (!strcmp(argv[i], "-dv"))
        {                                         // -dv option
            dataVStream.open(argv[++i], ios::in); // open data file
            if (!dataVStream)
            {
                cerr << "Cannot open data file\n";
                exit(1);
            }
            dataVals = &dataVStream; // make this the data stream
        }
        else if (!strcmp(argv[i], "-qf"))
        {                                         // -qf option
            queryStream.open(argv[++i], ios::in); // open query file
            if (!queryStream)
            {
                cerr << "Cannot open query file\n";
                exit(1);
            }
            queryIn = &queryStream; // make this query stream
        }
        else if (!strcmp(argv[i], "-af"))
        {                                            // -af option
            answersStream.open(argv[++i], ios::out); // open answers file
            if (!answersStream)
            {
                cerr << "Cannot open query file\n";
                exit(1);
            }
            answers = &answersStream; // make this query stream
        }
        else if (!strcmp(argv[i], "-test"))
        { // -test option
            string df("data/train/trainX");
            string dv("data/train/trainY");
            string qf("data/test/testX");
            string af("data/results/knn/results");
            string test(argv[++i]);

            df = df + test + ".pts";
            dv = dv + test + ".pts";
            qf = qf + test + ".pts";
            af = af + test + ".pts";

            //  data X file
            dataStream.open(df, ios::in);
            if (!dataStream)
            {
                cerr << "Cannot open data file\n";
                exit(1);
            }
            dataIn = &dataStream;

            //  data Y file
            dataVStream.open(dv, ios::in); // open data file
            if (!dataVStream)
            {
                cerr << "Cannot open data file\n";
                exit(1);
            }
            dataVals = &dataVStream;

            //  query file
            queryStream.open(qf, ios::in);
            if (!queryStream)
            {
                cerr << "Cannot open data file\n";
                exit(1);
            }
            queryIn = &queryStream;

            // output file
            answersStream.open(af, ios::out);
            if (!answersStream)
            {
                cerr << "Cannot open query file\n";
                exit(1);
            }
            answers = &answersStream;
        }
        else
        { // illegal syntax
            cerr << "Unrecognized option.\n";
            exit(1);
        }
        i++;
    }
    if (dataIn == NULL || queryIn == NULL)
    {
        cerr << "-df and -qf options must be specified\n";
        exit(1);
    }
}