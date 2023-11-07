#include <iostream>
#include <iomanip>
#include <sstream>
#include <utility>
#include <memory>
#include <ElasticRods/ElasticRod.hh>
#include <ElasticRods/python_bindings/visualization.hh>
#include <MeshFEM/newton_optimizer/newton_optimizer.hh>
#include "../ContactProblem.hh"
#include "../PeriodicRodList.hh"
#include "../SoftConstraint.hh"

#include <MeshFEM/GlobalBenchmark.hh>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
namespace py = pybind11;

using PyCallbackFunction = std::function<void(NewtonProblem *, size_t)>;
CallbackFunction callbackWrapper(const PyCallbackFunction &pcb) {
    if (pcb == nullptr) 
        return nullptr;
    else 
        return [pcb](NewtonProblem &p, size_t i) -> void { if (pcb) pcb(&p, i); };
}
using EnergyType = typename ElasticRod::EnergyType;

PYBIND11_MODULE(elastic_knots, m) {
    m.doc() = "Elastic Knots Codebase";

    py::module::import("sparse_matrices");

    py::class_<PeriodicRodList, std::shared_ptr<PeriodicRodList>>(m, "PeriodicRodList")
        .def(py::init<const std::vector<PeriodicRod> &>(), py::arg("rods"))
        .def(py::init<PeriodicRod &>(), py::arg("rod"))
        .def("__getitem__", [](const PeriodicRodList &prl, size_t i) -> std::shared_ptr<PeriodicRod> { return prl[i]; })  // equivalent to getRod
        .def("__iter__",    [](const PeriodicRodList &prl) { return py::make_iterator(prl.begin(), prl.end()); }, py::keep_alive<0, 1>())
        .def("getRod",              &PeriodicRodList::getRod, py::arg("rodIdx"), py::return_value_policy::reference_internal)
        .def("getRods",             &PeriodicRodList::getRods, py::return_value_policy::reference_internal)

        .def("size",                  &PeriodicRodList::size)
        .def("numRods",               &PeriodicRodList::numRods)
        .def("numEdges",              &PeriodicRodList::numEdges, py::arg("countGhost") = false)
        .def("numVertices",           &PeriodicRodList::numVertices, py::arg("countGhost") = false)
        .def("numDoF",                &PeriodicRodList::numDoF)
        .def("numEdgesPerRod",        &PeriodicRodList::numEdgesPerRod)
        .def("numVerticesPerRod",     &PeriodicRodList::numVerticesPerRod)
        .def("numDoFPerRod",          &PeriodicRodList::numDoFPerRod)
        .def("numEdgesInRod",         &PeriodicRodList::numEdgesInRod, py::arg("rodIdx"))
        .def("numVerticesInRod",      &PeriodicRodList::numVerticesInRod, py::arg("rodIdx"))
        .def("numDoFInRod",           &PeriodicRodList::numDoFInRod, py::arg("rodIdx"))
        .def("getDoFs",               &PeriodicRodList::getDoFs)
        .def("setDoFs",               &PeriodicRodList::setDoFs)
        .def("energy",                &PeriodicRodList::energy, py::arg("energyType") = EnergyType::Full)
        .def("gradient",              &PeriodicRodList::gradient, py::arg("updatedSource") = false)
        .def("hessian",               [](const PeriodicRodList &r) { SuiteSparseMatrix H = r.hessianSparsityPattern(); r.hessian(H); return H; })
        .def("hessianSparsityPattern", &PeriodicRodList::hessianSparsityPattern)
        .def("deformedPoints",        &PeriodicRodList::deformedPoints)
        .def("deformedLengths",       &PeriodicRodList::deformedLengths)
        .def("restLengths",           &PeriodicRodList::restLengths)

        .def("updateSourceFrame",                &PeriodicRodList::updateSourceFrame)
        .def("updateRotationParametrizations",   &PeriodicRodList::updateRotationParametrizations)

        .def("globalEdgeIndex",                   &PeriodicRodList::globalEdgeIndex, py::arg("li"), py::arg("rodIdx"))
        .def("globalNodeIndex",                   &PeriodicRodList::globalNodeIndex, py::arg("li"), py::arg("rodIdx"))
        .def("globalDofIndex",                    &PeriodicRodList::globalDofIndex, py::arg("dof"), py::arg("rodIdx"))
        .def("localEdgeIndex",                    &PeriodicRodList::localEdgeIndex, py::arg("gi"))
        .def("localNodeIndex",                    &PeriodicRodList::localNodeIndex, py::arg("gi"))
        .def("localDofIndex",                     &PeriodicRodList::localDofIndex, py::arg("dof"))
        .def("firstGlobalEdgeIndexInRod",         &PeriodicRodList::firstGlobalEdgeIndexInRod, py::arg("ri"))
        .def("firstGlobalNodeIndexInRod",         &PeriodicRodList::firstGlobalNodeIndexInRod, py::arg("ri"))
        .def("firstGlobalDofIndexInRod",          &PeriodicRodList::firstGlobalDofIndexInRod,  py::arg("ri"))
        .def("rodIndexFromGlobalEdgeIndex",       &PeriodicRodList::rodIndexFromGlobalEdgeIndex, py::arg("gi"))
        .def("rodIndexFromGlobalNodeIndex",       &PeriodicRodList::rodIndexFromGlobalNodeIndex, py::arg("gi"))
        .def("rodIndexFromGlobalDofIndex",        &PeriodicRodList::rodIndexFromGlobalDofIndex,  py::arg("dof"))
        .def("globalDofIndexFromGlobalNodeIndex", &PeriodicRodList::globalDofIndexFromGlobalNodeIndex, py::arg("gi"))
        .def("globalDofIndexFromGlobalEdgeIndex", &PeriodicRodList::globalDofIndexFromGlobalEdgeIndex, py::arg("gi"))
        .def("globalNodeIndexFromGlobalDofIndex", &PeriodicRodList::globalNodeIndexFromGlobalDofIndex, py::arg("dof"))

        .def("saveVisualizationGeometry",   &PeriodicRodList::saveVisualizationGeometry, py::arg("path"), py::arg("averagedMaterialFrames") = false)
        .def("visualizationGeometry",       &getVisualizationGeometry<PeriodicRodList>, py::arg("averagedMaterialFrames") = true)
        .def("visualizationField", [](const PeriodicRodList &r, const Eigen::VectorXd               &f) { return getVisualizationField(r, f); }, "Convert a per-vertex or per-edge field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))
        .def("visualizationField", [](const PeriodicRodList &r, const std::vector<Eigen::VectorXd>  &f) { return getVisualizationField(r, f); }, "Convert a per-vertex or per-edge field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))
        .def("visualizationField", [](const PeriodicRodList &r, const std::vector<Eigen::MatrixX3d> &f) { return getVisualizationField(r, f); }, "Convert a per-vertex or per-edge field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))
    ;

    // ----------------------------------------------------------------------------
    //                                  Problem
    // ----------------------------------------------------------------------------

    py::class_<ContactProblem, NewtonProblem>(m, "ContactProblem")
        .def(py::init<PeriodicRodList &, ContactProblemOptions>(), 
            py::arg("rods"), py::arg("problemOptions"))
        .def("numRods",                   &ContactProblem::numRods)
        .def("numIPCConstraints",         &ContactProblem::numIPCConstraints)
        .def("getVars",                   &ContactProblem::getVars)
        .def("numVars",                   &ContactProblem::numVars)
        .def("getDoFs",                   &ContactProblem::getVars)                    // alias
        .def("setVars",                   &ContactProblem::setVars, py::arg("vars"))
        .def("setDoFs",                   &ContactProblem::setVars, py::arg("vars"))   // alias
        .def("addSoftConstraint",         &ContactProblem::addSoftConstraint,  py::arg("softConstraint"))
        .def("addSoftConstraints",        &ContactProblem::addSoftConstraints, py::arg("softConstraints"))
        .def("hasCollisions",             &ContactProblem::hasCollisions)
        .def("updateConstraintSet",       &ContactProblem::updateConstraintSet)
        .def("contactEnergy",             &ContactProblem::contactEnergy)
        .def("externalPotentialEnergy",   &ContactProblem::externalPotentialEnergy)
        .def("hessianSparsityPattern",    &ContactProblem::hessianSparsityPattern)
        .def("hessian",                   &ContactProblem::hessian)
        .def("contactForces",             &ContactProblem::contactForces)
        .def_readwrite("externalForces",  &ContactProblem::external_forces)
        .def_readwrite("options",         &ContactProblem::m_options)
        .def_readwrite("constraintSet",   &ContactProblem::m_constraintSet)
        .def_readwrite("collisionMesh",   &ContactProblem::m_collisionMesh)
    ;

    // ----------------------------------------------------------------------------
    //                                  Soft Constraints
    // ----------------------------------------------------------------------------

    py::class_<SoftConstraint, std::shared_ptr<SoftConstraint>>(m, "SoftConstraint");

    py::class_<FlatnessConstraint, SoftConstraint, std::shared_ptr<FlatnessConstraint>>(m, "FlatnessConstraint")
        .def(py::init<Real, const Eigen::Vector3d &, const Eigen::Vector3d &, Real, Real>(), 
            py::arg("stiffness"), py::arg("n"), py::arg("center"), py::arg("upper_d"), py::arg("lower_d"))
        .def("energy", [](const FlatnessConstraint &fc, const PeriodicRodList &rodsList) {
            Real energy_value = 0.0;
            fc.energy(rodsList, energy_value);
            return energy_value;
        })
        .def("gradient", [](const FlatnessConstraint &fc, const PeriodicRodList &rodsList) {
            Eigen::VectorXd g;
            g.setZero(rodsList.numVertices() * 3);
            fc.gradient(rodsList, g);
            return g;
        })
        ;

    py::class_<VolumeConstraint, SoftConstraint, std::shared_ptr<VolumeConstraint>>(m, "VolumeConstraint")
        .def(py::init<Real, const Eigen::Vector3d &>(), py::arg("stiffness"), py::arg("aspectRatio"))
        .def(py::init<Real>(), py::arg("stiffness"))
        ;

    py::class_<SphericalShellConstraint, SoftConstraint, std::shared_ptr<SphericalShellConstraint>>(m, "SphericalShellConstraint")
        .def(py::init<Real, const Eigen::Vector3d &, Real, Real>(), py::arg("stiffness"), py::arg("center"), py::arg("upper_d"), py::arg("lower_d"))
        .def(py::init<Real>(), py::arg("stiffness"))
        ;

    // ----------------------------------------------------------------------------
    //                                  Options
    // ----------------------------------------------------------------------------

    py::class_<ContactProblemOptions>(m, "ContactProblemOptions")
        .def(py::init<>())
        .def_readwrite("hasCollisions",             &ContactProblemOptions::hasCollisions)
        .def_readwrite("printIterInfo",             &ContactProblemOptions::printIterInfo)
        .def_readwrite("minContactEdgeDist",        &ContactProblemOptions::minContactEdgeDist)
        .def_readwrite("Wang2021MaxIter",           &ContactProblemOptions::Wang2021MaxIter)
        .def_readwrite("projectContactHessianPSD",  &ContactProblemOptions::projectContactHessianPSD)
        .def_readwrite("contactStiffness",          &ContactProblemOptions::contactStiffness)
        .def_readwrite("dHat",                      &ContactProblemOptions::dHat)
    ;

    // ----------------------------------------------------------------------------
    //                                 Utilities
    // ----------------------------------------------------------------------------

    m.def("minimize_twist",
        [](PeriodicRod &pr, bool verbose) {
            py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
            py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
            return minimize_twist(pr, verbose);
        },
        py::arg("pr"),
        py::arg("verbose") = false
    );

    m.def("spread_twist_preserving_link",
        [](PeriodicRod &pr, bool verbose) {
            py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
            py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
            return spread_twist_preserving_link(pr, verbose);
        },
        py::arg("pr"),
        py::arg("verbose") = false
    );

    m.def("compute_equilibrium",
        [](
            PeriodicRodList rods,
            const ContactProblemOptions &problemOptions, 
            const NewtonOptimizerOptions &optimizerOptions, 
            const std::vector<size_t> &fixedVars,
            const Eigen::VectorXd &externalForces,
            const ContactProblem::SoftConstraintsList &softConstraints,
            const PyCallbackFunction &pcb,
            double hessianShift
        ) {
            py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
            py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
            auto cb = callbackWrapper(pcb);
            return compute_equilibrium(rods, problemOptions, optimizerOptions, fixedVars, externalForces, softConstraints, cb, hessianShift);
        },
        py::arg("rods"),
        py::arg("problemOptions") = ContactProblemOptions(),
        py::arg("optimizerOptions") = NewtonOptimizerOptions(),
        py::arg("fixedVars") = std::vector<size_t>(),
        py::arg("externalForces") = Eigen::VectorXd(),
        py::arg("softConstraints") = ContactProblem::SoftConstraintsList(),
        py::arg("callback") = nullptr,
        py::arg("hessianShift") = 0.0
    );

}
