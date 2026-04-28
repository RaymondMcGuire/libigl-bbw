/*
 * File: main.cpp
 * Module: 403_BoundedBiharmonicWeights
 * Created Date: 2026-04-28
 * Author: Xu WANG
 * -----
 * Last Modified: 2026-04-28
 * Modified By: Xu WANG
 * -----
 * Copyright (c) 2026 Xu WANG
 */

#include <igl/boundary_conditions.h>
#include <igl/boundary_facets.h>
#include <igl/colon.h>
#include <igl/column_to_quats.h>
#include <igl/directed_edge_parents.h>
#include <igl/forward_kinematics.h>
#include <igl/jet.h>
#include <igl/lbs_matrix.h>
#include <igl/deform_skeleton.h>
#include <igl/readTGF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/bbw.h>

#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <vector>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>

typedef std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>>
    RotationList;

const Eigen::RowVector3d sea_green(70. / 255., 252. / 255., 167. / 255.);
int selected = 0;
Eigen::MatrixXd V, W, U, C, M;
Eigen::MatrixXi T, F, BE;
Eigen::VectorXi P;
RotationList pose;
double anim_t = 1.0;
double anim_t_dir = -0.03;
const bool snap_skeleton_to_tet_vertices = true;
Eigen::MatrixXd handle_colors;

Eigen::RowVector3d hsv_to_rgb(double h, const double s, const double v)
{
  h -= std::floor(h);
  const double c = v * s;
  const double x = c * (1.0 - std::abs(std::fmod(h * 6.0, 2.0) - 1.0));
  const double m = v - c;
  Eigen::RowVector3d rgb(0.0, 0.0, 0.0);

  if (h < 1.0 / 6.0)
  {
    rgb << c, x, 0.0;
  }
  else if (h < 2.0 / 6.0)
  {
    rgb << x, c, 0.0;
  }
  else if (h < 3.0 / 6.0)
  {
    rgb << 0.0, c, x;
  }
  else if (h < 4.0 / 6.0)
  {
    rgb << 0.0, x, c;
  }
  else if (h < 5.0 / 6.0)
  {
    rgb << x, 0.0, c;
  }
  else
  {
    rgb << c, 0.0, x;
  }

  return rgb.array() + m;
}

void ensure_handle_colors()
{
  if (handle_colors.rows() == W.cols() && handle_colors.cols() == 3)
  {
    return;
  }

  handle_colors.resize(W.cols(), 3);
  for (int i = 0; i < W.cols(); i++)
  {
    const double hue = std::fmod(0.07 + i * 0.6180339887498949, 1.0);
    handle_colors.row(i) = hsv_to_rgb(hue, 0.68, 0.95);
  }
}

double surface_weight_range(const int column)
{
  if (F.rows() == 0 || W.rows() == 0 || column < 0 || column >= W.cols())
  {
    return 0.0;
  }

  double min_value = std::numeric_limits<double>::infinity();
  double max_value = -std::numeric_limits<double>::infinity();
  for (int f = 0; f < F.rows(); f++)
  {
    for (int c = 0; c < F.cols(); c++)
    {
      const int v = F(f, c);
      if (v >= 0 && v < W.rows())
      {
        const double w = W(v, column);
        min_value = std::min(min_value, w);
        max_value = std::max(max_value, w);
      }
    }
  }

  return std::isfinite(min_value) && std::isfinite(max_value)
             ? max_value - min_value
             : 0.0;
}

int next_visible_weight_column(const int start, const int direction)
{
  if (W.cols() == 0)
  {
    return 0;
  }

  int column = start;
  for (int k = 0; k < W.cols(); k++)
  {
    column = (column + direction + W.cols()) % W.cols();
    if (surface_weight_range(column) > 1e-8)
    {
      return column;
    }
  }

  return std::min(std::max(start, 0), (int)W.cols() - 1);
}

void set_weight_column(igl::opengl::glfw::Viewer &viewer, const int column)
{
  selected = std::min(std::max(column, 0), (int)W.cols() - 1);
  viewer.data().set_data(W.col(selected));
  std::cout << "Showing weight column " << selected << " / "
            << W.cols() - 1 << " for bone edge " << selected
            << " (surface range " << surface_weight_range(selected)
            << ")." << std::endl;
}

void set_all_weights_blend(igl::opengl::glfw::Viewer &viewer)
{
  if (W.rows() == 0 || W.cols() == 0)
  {
    return;
  }

  ensure_handle_colors();
  Eigen::MatrixXd colors = W * handle_colors;
  colors = colors.cwiseMax(0.0).cwiseMin(1.0);
  viewer.data().show_texture = false;
  viewer.data().set_colors(colors);
  std::cout << "Showing all weights as blended handle colors. "
            << "Each vertex color is sum_i W(v,i) * handle_color(i)."
            << std::endl;
}

void set_dominant_weight_map(igl::opengl::glfw::Viewer &viewer)
{
  if (W.rows() == 0 || W.cols() == 0)
  {
    return;
  }

  ensure_handle_colors();
  Eigen::MatrixXd colors(W.rows(), 3);
  for (int v = 0; v < W.rows(); v++)
  {
    int dominant = 0;
    double max_weight = W(v, 0);
    for (int h = 1; h < W.cols(); h++)
    {
      if (W(v, h) > max_weight)
      {
        max_weight = W(v, h);
        dominant = h;
      }
    }
    colors.row(v) = handle_colors.row(dominant);
  }

  viewer.data().show_texture = false;
  viewer.data().set_colors(colors);
  std::cout << "Showing dominant weight map. Each vertex uses the color of "
            << "the bone edge with max W(v,i)." << std::endl;
}

std::vector<int> snap_control_points_to_tet_vertices(
    const Eigen::MatrixXd &vertices,
    Eigen::MatrixXd &controls)
{
  std::vector<int> nearest_vertex(controls.rows(), -1);
  if (vertices.rows() == 0 || controls.rows() == 0)
  {
    return nearest_vertex;
  }

  std::vector<bool> used(vertices.rows(), false);
  double max_distance = 0.0;
  double sum_distance = 0.0;
  int reused_vertices = 0;

  for (int c = 0; c < controls.rows(); c++)
  {
    int best = -1;
    double best_squared_distance = std::numeric_limits<double>::infinity();
    for (int v = 0; v < vertices.rows(); v++)
    {
      if (used[v])
      {
        continue;
      }
      const double squared_distance =
          (vertices.row(v) - controls.row(c)).squaredNorm();
      if (squared_distance < best_squared_distance)
      {
        best_squared_distance = squared_distance;
        best = v;
      }
    }

    if (best < 0)
    {
      reused_vertices++;
      for (int v = 0; v < vertices.rows(); v++)
      {
        const double squared_distance =
            (vertices.row(v) - controls.row(c)).squaredNorm();
        if (squared_distance < best_squared_distance)
        {
          best_squared_distance = squared_distance;
          best = v;
        }
      }
    }
    else
    {
      used[best] = true;
    }

    nearest_vertex[c] = best;
    controls.row(c) = vertices.row(best);
    const double distance = std::sqrt(best_squared_distance);
    max_distance = std::max(max_distance, distance);
    sum_distance += distance;
  }

  std::cout << "Snapped " << controls.rows()
            << " skeleton control points to tet vertices. Average move: "
            << (sum_distance / std::max(1, (int)controls.rows()))
            << ", max move: " << max_distance << "." << std::endl;
  if (reused_vertices > 0)
  {
    std::cout << "Warning: " << reused_vertices
              << " control points had to reuse a tet vertex because there "
              << "were more controls than available unique vertices."
              << std::endl;
  }
  return nearest_vertex;
}

void report_degenerate_bone_edges(
    const Eigen::MatrixXi &bone_edges,
    const std::vector<int> &nearest_vertex)
{
  int degenerate_edges = 0;
  for (int e = 0; e < bone_edges.rows(); e++)
  {
    const int a = bone_edges(e, 0);
    const int b = bone_edges(e, 1);
    if (a >= 0 && a < (int)nearest_vertex.size() &&
        b >= 0 && b < (int)nearest_vertex.size() &&
        nearest_vertex[a] == nearest_vertex[b])
    {
      degenerate_edges++;
    }
  }

  if (degenerate_edges > 0)
  {
    std::cout << "Warning: " << degenerate_edges
              << " bone edges still have both endpoints on the same tet "
              << "vertex after snapping." << std::endl;
  }
}

template <typename DerivedM>
void write_json_matrix(
    std::ostream &out,
    const char *name,
    const Eigen::MatrixBase<DerivedM> &M,
    const int index_offset = 0)
{
  out << "  \"" << name << "\": [\n";
  for (int i = 0; i < M.rows(); i++)
  {
    out << "    [";
    for (int j = 0; j < M.cols(); j++)
    {
      if (j > 0)
      {
        out << ", ";
      }
      out << std::setprecision(17) << (M(i, j) + index_offset);
    }
    out << "]";
    if (i + 1 < M.rows())
    {
      out << ",";
    }
    out << "\n";
  }
  out << "  ]";
}

bool write_bbw_json(
    const std::string &file_name,
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &T,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &W,
    const Eigen::MatrixXd &C,
    const Eigen::MatrixXi &BE)
{
  std::ofstream out(file_name);
  if (!out)
  {
    return false;
  }

  out << "{\n";
  out << "  \"format\": \"libigl_bbw_export\",\n";
  out << "  \"index_base\": 1,\n";
  out << "  \"weights_columns\": \"bone_edges\",\n";
  write_json_matrix(out, "vertices", V);
  out << ",\n";
  write_json_matrix(out, "tetrahedra", T, 1);
  out << ",\n";
  write_json_matrix(out, "faces", F, 1);
  out << ",\n";
  write_json_matrix(out, "weights", W);
  out << ",\n";
  write_json_matrix(out, "joints", C);
  out << ",\n";
  write_json_matrix(out, "bone_edges", BE, 1);
  out << "\n}\n";

  return true;
}

bool normalize_indices(
    Eigen::MatrixXi &I,
    const int vertex_count,
    const std::string &name)
{
  if (I.size() == 0)
  {
    return true;
  }

  int min_index = I.minCoeff();
  int max_index = I.maxCoeff();
  if (min_index >= 1 && max_index == vertex_count)
  {
    I.array() -= 1;
    min_index -= 1;
    max_index -= 1;
  }

  if (min_index < 0 || max_index >= vertex_count)
  {
    std::cerr << name << " indices are out of range: [" << min_index
              << ", " << max_index << "] for " << vertex_count
              << " vertices." << std::endl;
    return false;
  }

  return true;
}

bool read_node_file(const std::string &file_name, Eigen::MatrixXd &V)
{
  std::ifstream in(file_name);
  if (!in)
  {
    return false;
  }

  std::string line;
  if (!std::getline(in, line))
  {
    return false;
  }

  std::istringstream header(line);
  int vertex_count = 0;
  int dim = 0;
  int attribute_count = 0;
  int marker_count = 0;
  if (!(header >> vertex_count >> dim >> attribute_count >> marker_count) ||
      vertex_count < 0 || dim < 3)
  {
    return false;
  }

  std::vector<int> ids;
  std::vector<Eigen::RowVector3d> points;
  ids.reserve(vertex_count);
  points.reserve(vertex_count);
  int min_id = std::numeric_limits<int>::max();
  int max_id = std::numeric_limits<int>::min();

  while ((int)ids.size() < vertex_count && std::getline(in, line))
  {
    if (line.empty() || line[0] == '#')
    {
      continue;
    }

    std::istringstream row(line);
    int id = 0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    if (!(row >> id >> x >> y >> z))
    {
      return false;
    }
    ids.push_back(id);
    points.emplace_back(x, y, z);
    min_id = std::min(min_id, id);
    max_id = std::max(max_id, id);
  }

  if ((int)ids.size() != vertex_count)
  {
    return false;
  }

  int index_offset = 0;
  if (min_id == 1 && max_id == vertex_count)
  {
    index_offset = 1;
  }
  else if (!(min_id == 0 && max_id == vertex_count - 1))
  {
    std::cerr << "Unsupported .node vertex ids: [" << min_id << ", "
              << max_id << "]. Expected contiguous 0-based or 1-based ids."
              << std::endl;
    return false;
  }

  V.resize(vertex_count, 3);
  for (int i = 0; i < vertex_count; i++)
  {
    const int row_index = ids[i] - index_offset;
    if (row_index < 0 || row_index >= vertex_count)
    {
      return false;
    }
    V.row(row_index) = points[i];
  }

  return true;
}

bool read_ele_file(
    const std::string &file_name,
    const int vertex_count,
    Eigen::MatrixXi &T)
{
  std::ifstream in(file_name);
  if (!in)
  {
    return false;
  }

  std::string line;
  if (!std::getline(in, line))
  {
    return false;
  }

  std::istringstream header(line);
  int tet_count = 0;
  int nodes_per_tet = 0;
  int attribute_count = 0;
  if (!(header >> tet_count >> nodes_per_tet >> attribute_count) ||
      tet_count < 0 || nodes_per_tet != 4)
  {
    return false;
  }

  T.resize(tet_count, 4);
  int row_index = 0;
  while (row_index < tet_count && std::getline(in, line))
  {
    if (line.empty() || line[0] == '#')
    {
      continue;
    }

    std::istringstream row(line);
    int id = 0;
    if (!(row >> id >> T(row_index, 0) >> T(row_index, 1) >>
          T(row_index, 2) >> T(row_index, 3)))
    {
      T.resize(0, 4);
      return false;
    }
    row_index++;
  }

  if (row_index != tet_count)
  {
    T.resize(0, 4);
    return false;
  }

  return normalize_indices(T, vertex_count, ".ele");
}

bool read_face_file(
    const std::string &file_name,
    const int vertex_count,
    Eigen::MatrixXi &F)
{
  std::ifstream in(file_name);
  if (!in)
  {
    return false;
  }

  std::string line;
  if (!std::getline(in, line))
  {
    return false;
  }

  int face_count = 0;
  {
    std::istringstream header(line);
    if (!(header >> face_count) || face_count < 0)
    {
      return false;
    }
  }

  F.resize(face_count, 3);
  int row_index = 0;
  while (row_index < face_count && std::getline(in, line))
  {
    if (line.empty() || line[0] == '#')
    {
      continue;
    }

    std::istringstream row(line);
    std::vector<int> values;
    int value = 0;
    while (row >> value)
    {
      values.push_back(value);
    }
    if (values.size() < 3)
    {
      F.resize(0, 3);
      return false;
    }

    const bool has_face_id =
        values.size() >= 4 &&
        (values[0] == row_index || values[0] == row_index + 1);
    const int first_vertex = has_face_id ? 1 : 0;
    F(row_index, 0) = values[first_vertex + 0];
    F(row_index, 1) = values[first_vertex + 1];
    F(row_index, 2) = values[first_vertex + 2];
    row_index++;
  }

  if (row_index != face_count)
  {
    F.resize(0, 3);
    return false;
  }

  return normalize_indices(F, vertex_count, ".face");
}

bool pre_draw(igl::opengl::glfw::Viewer &viewer)
{
  using namespace Eigen;
  using namespace std;
  if (viewer.core().is_animating)
  {
    // Interpolate pose and identity
    RotationList anim_pose(pose.size());
    for (int e = 0; e < pose.size(); e++)
    {
      anim_pose[e] = pose[e].slerp(anim_t, Quaterniond::Identity());
    }
    // Propagate relative rotations via FK to retrieve absolute transformations
    RotationList vQ;
    vector<Vector3d> vT;
    igl::forward_kinematics(C, BE, P, anim_pose, vQ, vT);
    const int dim = C.cols();
    MatrixXd T(BE.rows() * (dim + 1), dim);
    for (int e = 0; e < BE.rows(); e++)
    {
      Affine3d a = Affine3d::Identity();
      a.translate(vT[e]);
      a.rotate(vQ[e]);
      T.block(e * (dim + 1), 0, dim + 1, dim) =
          a.matrix().transpose().block(0, 0, dim + 1, dim);
    }
    // Compute deformation via LBS as matrix multiplication
    U = M * T;

    // Also deform skeleton edges
    MatrixXd CT;
    MatrixXi BET;
    igl::deform_skeleton(C, BE, T, CT, BET);

    viewer.data().set_vertices(U);
    viewer.data().set_edges(CT, BET, sea_green);
    viewer.data().compute_normals();
    anim_t += anim_t_dir;
    anim_t_dir *= (anim_t >= 1.0 || anim_t <= 0.0 ? -1.0 : 1.0);
  }
  return false;
}

bool key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int mods)
{
  switch (key)
  {
  case ' ':
    viewer.core().is_animating = !viewer.core().is_animating;
    break;
  case '.':
    set_weight_column(viewer, next_visible_weight_column(selected, 1));
    break;
  case ',':
    set_weight_column(viewer, next_visible_weight_column(selected, -1));
    break;
  case 'a':
  case 'A':
    set_all_weights_blend(viewer);
    break;
  case 'd':
  case 'D':
    set_dominant_weight_map(viewer);
    break;
  }
  return true;
}

int main(int argc, char *argv[])
{
  using namespace Eigen;
  using namespace std;

  const std::string output_dir = "E:/data/cgi2026-liu-data-submit/libigl-bbw/output";
  const std::string input_dir = "E:/data/cgi2026-liu-data-submit/libigl-bbw/assets";

  const std::string input_prefix =
      input_dir + "/body_mesh";
  if (!read_node_file(input_prefix + ".node", V))
  {
    cerr << "Could not read " << input_prefix << ".node" << endl;
    return EXIT_FAILURE;
  }
  if (!read_ele_file(input_prefix + ".ele", V.rows(), T))
  {
    cerr << "Could not read " << input_prefix << ".ele" << endl;
    return EXIT_FAILURE;
  }
  U = V;
  if (read_face_file(input_prefix + ".face", V.rows(), F))
  {
    cout << "Loaded " << V.rows() << " vertices, " << T.rows()
         << " tetrahedra and " << F.rows() << " faces from TetGen files."
         << endl;
  }
  else if (T.rows() > 0)
  {
    igl::boundary_facets(T, F);
    cout << "Could not read " << input_prefix
         << ".face; extracted " << F.rows()
         << " boundary faces from tetrahedra for display/export." << endl;
  }
  if (!igl::readTGF(input_prefix + ".tgf", C, BE))
  {
    cerr << "Could not read " << input_prefix << ".tgf" << endl;
    return EXIT_FAILURE;
  }
  if (snap_skeleton_to_tet_vertices)
  {
    const std::vector<int> nearest_vertex =
        snap_control_points_to_tet_vertices(V, C);
    report_degenerate_bone_edges(BE, nearest_vertex);
  }
  // retrieve parents for forward kinematics
  igl::directed_edge_parents(BE, P);

  // A pose file is not needed to compute BBW weights. Use identity rotations
  // so the viewer can still run without an animation/pose file.
  pose.resize(BE.rows(), Eigen::Quaterniond::Identity());

  // List of boundary indices (aka fixed value indices into VV)
  VectorXi b;
  // List of boundary conditions of each weight function
  MatrixXd bc;
  const bool boundary_conditions_ok =
      igl::boundary_conditions(V, T, C, VectorXi(), BE, MatrixXi(), MatrixXi(), b, bc);
  int constrained_handles = 0;
  for (int i = 0; i < bc.cols(); i++)
  {
    if (bc.col(i).maxCoeff() > 1.0 - 1e-7)
    {
      constrained_handles++;
    }
  }
  cout << "Boundary conditions: " << b.rows() << " constrained vertices, "
       << constrained_handles << " / " << bc.cols()
       << " handles receive a full constraint." << endl;
  if (!boundary_conditions_ok)
  {
    cerr << "Warning: boundary_conditions reported suspicious constraints. "
         << "With nearest-vertex snapping, this can happen because internal "
         << "bone endpoints are shared by multiple bones and no extra samples "
         << "exist along each bone segment." << endl;
  }

  // compute BBW weights matrix
  igl::BBWData bbw_data;
  // only a few iterations for sake of demo
  bbw_data.active_set_params.max_iter = 8;
  bbw_data.verbosity = 2;
  if (!igl::bbw(V, T, b, bc, bbw_data, W))
  {
    return EXIT_FAILURE;
  }

  // MatrixXd Vsurf = V.topLeftCorner(F.maxCoeff()+1,V.cols());
  // MatrixXd Wsurf;
  // if(!igl::bone_heat(Vsurf,F,C,VectorXi(),BE,MatrixXi(),Wsurf))
  //{
  //   return false;
  // }
  // W.setConstant(V.rows(),Wsurf.cols(),1);
  // W.topLeftCorner(Wsurf.rows(),Wsurf.cols()) = Wsurf = Wsurf = Wsurf = Wsurf;

  // Normalize weights to sum to one
  W = (W.array().colwise() / W.array().rowwise().sum()).eval();

  int visible_weight_columns = 0;
  for (int i = 0; i < W.cols(); i++)
  {
    if (surface_weight_range(i) > 1e-8)
    {
      visible_weight_columns++;
    }
  }
  selected = next_visible_weight_column(-1, 1);
  cout << "Visible weight columns on the surface: " << visible_weight_columns
       << " / " << W.cols() << ". Initial column: " << selected << "." << endl;

  // Export data for MATLAB inspection. W rows correspond to V rows, and W
  // columns correspond to the bone edges in BE.

  std::error_code create_dir_error;
  std::filesystem::create_directories(output_dir, create_dir_error);
  if (create_dir_error)
  {
    cerr << "Could not create output directory: " << output_dir << endl;
    return EXIT_FAILURE;
  }
  const std::string output_json = output_dir + "/body-bbw.json";
  if (!write_bbw_json(output_json, V, T, F, W, C, BE))
  {
    cerr << "Could not write " << output_json << endl;
    return EXIT_FAILURE;
  }
  cout << "Wrote " << output_json << endl;

  // precompute linear blend skinning matrix
  igl::lbs_matrix(V, W, M);

  // Plot the mesh with pseudocolors
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(U, F);
  set_weight_column(viewer, selected);
  viewer.data().set_edges(C, BE, sea_green);
  viewer.data().show_lines = false;
  viewer.data().show_overlay_depth = false;
  viewer.data().line_width = 1;
  viewer.callback_pre_draw = &pre_draw;
  viewer.callback_key_down = &key_down;
  viewer.core().is_animating = false;
  viewer.core().animation_max_fps = 30.;
  cout << "Press '.' to show next weight function." << endl
       << "Press ',' to show previous weight function." << endl
       << "Press 'A' to show all weights as blended colors." << endl
       << "Press 'D' to show the dominant weight map." << endl
       << "Press [space] to toggle animation." << endl;
  viewer.launch();
  return EXIT_SUCCESS;
}
