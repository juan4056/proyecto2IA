#include "node.hpp"

typedef vector<vector<float>> df_float;
typedef vector<float> vfloat;

class tree
{
public:
  node *root;
  int classes, features;

public:
  tree(int _classes, int f) : classes(_classes), features(f)
  {
    root = nullptr;
  }

  void find_max_min(df_float data, vfloat &max_v, vfloat &min_v, vfloat &mean_v)
  {
    max_v.assign(features, 0);
    min_v.assign(features, 100);
    for (int i = 0; i < data.size(); i++)
    {
      for (int j = 0; j < features; j++)
      {
        if (data[i][j] > max_v[j])
          max_v[j] = data[i][j];
        if (data[i][j] < min_v[j])
          min_v[j] = data[i][j];
      }
    }
    for (int i = 0; i < features; i++)
    {
      mean_v.push_back((max_v[i] + min_v[i]) / 2);
    }
  }

  float gini_impurity(df_float data)
  {
    vfloat probabilities(classes + 1, 0);
    float gi = 0;

    for (int i = 0; i < data.size(); i++)
      probabilities[data[i].back()]++;

    for (int i = 0; i < probabilities.size(); i++)
      probabilities[i] = probabilities[i] / data.size();

    for (int i = 0; i < probabilities.size(); i++)
      gi += probabilities[i] * (1 - probabilities[i]);

    return gi;
  }

  float gini_gain(df_float data, float f, float splt)
  {
    df_float data_l, data_r;
    for (int i = 0; i < data.size(); i++)
    {
      if (data[i][f] <= splt)
        data_l.push_back(data[i]);
      else
        data_r.push_back(data[i]);
    }

    float gi_data = gini_impurity(data), gi_l = gini_impurity(data_l),
          gi_r = gini_impurity(data_r);

    float pl = (data_l.size() / data.size()) * gi_l;
    float pr = (data_r.size() / data.size()) * gi_r;

    return gi_data - (pl + pr);
  }

  void split_node(node *nd, int f, float split)
  {
    if (nd->is_leaf)
    {
      df_float data_l, data_r, df_float, data = nd->data;
      for (int i = 0; i < data.size(); i++)
      {
        if (data[i][f] <= split)
          data_l.push_back(data[i]);
        else
          data_r.push_back(data[i]);
      }
      node *node_l = new node{nullptr, nullptr, true, 0, 0, data_l},
           *node_r = new node{nullptr, nullptr, true, 0, 0, data_r};

      nd->is_leaf = false;
      nd->left = node_l;
      nd->right = node_r;
      nd->feature = f;
      nd->split = split;
    }
  }

  void build_node(node *nd)
  {

    if (gini_impurity(nd->data) == 0)
      return;

    vfloat max_v, min_v, mean_v;
    find_max_min(nd->data, max_v, min_v, mean_v);

    float it, f_select = -1;
    float b_splt = 0, max_gg = -10;
    for (int i = 0; i < features; i++)
    {
      it = 0.05;
      for (float t_s = min_v[i]; t_s <= max_v[i];
           t_s += it)
      {
        float gg = gini_gain(nd->data, i, t_s);
        if (gg > max_gg)
        {
          max_gg = gg;
          b_splt = t_s;
          f_select = i;
        }
      }
    }
    split_node(nd, f_select, b_splt);
    build_node(nd->left);
    build_node(nd->right);
  }

  void build(df_float data)
  {
    root = new node{nullptr, nullptr, true, 0, 0, data};
    build_node(root);
  }

  void print(node *nd)
  {
    if (nd->is_leaf)
    {
      cout << nd->data.size() << "\t" << nd->split << "\t" << nd->feature << endl;
      for (int i = 0; i < nd->data.size(); i++)
        cout << nd->data[i].back() << " ";
      cout << endl;
    }
    if (nd->left)
      print(nd->left);
    if (nd->right)
      print(nd->right);
  }
  int predict(vfloat data)
  {
    node *temp = root;
    while (!temp->is_leaf)
    {
      if (data[temp->feature] <= temp->split)
        temp = temp->left;
      else
        temp = temp->right;
    }
    return temp->data[0].back();
  }
};