#include "distance_function.h"

using namespace model;
using namespace dist_functions;

double EuklideanSquaredDist::operator()(Vector x, Vector y) {
    return (x - y).squaredNorm();
}

Vector EuklideanSquaredDist::GetGradientX(Vector x, Vector y) {
    return 2 * x - 2 * y;
}
