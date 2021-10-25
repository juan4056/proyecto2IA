#include <iostream>
#include <vector>
using namespace std;

struct node
{
  node *left;
  node *right;
  bool is_leaf;
  int feature;
  float split;
  vector<vector<float>> data;
};