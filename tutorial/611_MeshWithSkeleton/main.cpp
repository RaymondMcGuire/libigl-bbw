#include <igl/copyleft/tetgen/mesh_with_skeleton.h>
#include <igl/readTGF.h>
#include <igl/read_triangle_mesh.h>
#include <igl/writeMESH.h>

#include <Eigen/Core>

#include <iostream>
#include <string>

namespace
{
const std::string kObjPath =
    "C:/Users/localhost/Downloads/data/data/body_mesh.obj";
const std::string kTgfPath =
    std::string(TUTORIAL_SHARED_PATH) + "/body_mesh.tgf";
const std::string kOutputMeshPath =
    "E:/data/cgi2026-liu/libigl/output/body_skeleton.mesh";
const int kSamplesPerBone = 10;
const std::string kTetgenFlags = "pq2Y";

bool read_skeleton_as_bones(
    const std::string &tgf_path,
    Eigen::MatrixXd &C,
    Eigen::VectorXi &P,
    Eigen::MatrixXi &BE,
    Eigen::MatrixXi &CE)
{
  Eigen::MatrixXi E;
  Eigen::MatrixXi PE;
  if (!igl::readTGF(tgf_path, C, E, P, BE, CE, PE))
  {
    return false;
  }

  if (BE.rows() == 0 && E.rows() > 0)
  {
    // Simple .tgf files often store skeleton edges as just "source target"
    // without the optional "is bone" flag. Treat those edges as bones.
    BE = E;
    P.resize(0);
    CE.resize(0, 2);
  }

  return true;
}
}

int main()
{
  using namespace Eigen;
  using namespace std;

  if (kSamplesPerBone < 0)
  {
    cerr << "samples_per_bone must be non-negative." << endl;
    return EXIT_FAILURE;
  }

  MatrixXd V;
  MatrixXi F;
  if (!igl::read_triangle_mesh(kObjPath, V, F))
  {
    cerr << "Could not read surface mesh: " << kObjPath << endl;
    return EXIT_FAILURE;
  }

  MatrixXd C;
  VectorXi P;
  MatrixXi BE;
  MatrixXi CE;
  if (!read_skeleton_as_bones(kTgfPath, C, P, BE, CE))
  {
    cerr << "Could not read skeleton graph: " << kTgfPath << endl;
    return EXIT_FAILURE;
  }

  if (BE.rows() == 0 && P.rows() == 0 && CE.rows() == 0)
  {
    cerr << "The TGF file contains no point, bone, or cage handles." << endl;
    return EXIT_FAILURE;
  }

  cout << "Surface mesh: " << V.rows() << " vertices, " << F.rows()
       << " faces." << endl;
  cout << "Skeleton graph: " << C.rows() << " joints/control points, "
       << BE.rows() << " bone edges, " << P.rows() << " point handles."
       << endl;
  cout << "TetGen flags: "
       << (kTetgenFlags.empty() ? "default mesh_with_skeleton flags" : kTetgenFlags)
       << ", samples per bone: " << kSamplesPerBone << "." << endl;

  MatrixXd TV;
  MatrixXi TT;
  MatrixXi TF;
  const bool ok = igl::copyleft::tetgen::mesh_with_skeleton(
      V,
      F,
      C,
      P,
      BE,
      CE,
      kSamplesPerBone,
      kTetgenFlags,
      TV,
      TT,
      TF);

  if (!ok)
  {
    cerr << "TetGen failed to create a skeleton-conforming tetrahedral mesh."
         << endl;
    return EXIT_FAILURE;
  }

  if (!igl::writeMESH(kOutputMeshPath, TV, TT, TF))
  {
    cerr << "Could not write output mesh: " << kOutputMeshPath << endl;
    return EXIT_FAILURE;
  }

  cout << "Wrote " << kOutputMeshPath << " with " << TV.rows() << " vertices, "
       << TT.rows() << " tetrahedra, and " << TF.rows() << " boundary faces."
       << endl;
  return EXIT_SUCCESS;
}
