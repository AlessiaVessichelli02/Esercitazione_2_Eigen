#include <iostream>
#include <iomanip>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;

void SystemSolution(const MatrixXd& A, const VectorXd& b, const VectorXd& c)
{
    //PALU
    PartialPivLU<MatrixXd> lu(A);
    Vector2d x_PALU = lu.solve(b);

    //QR
    HouseholderQR<MatrixXd> qr(A);
    Vector2d x_QR = qr.solve(b);

    //Stampo i risultati
    //cout << "Solution using PALU Decomposition:\n" << fixed << setprecision(5) << scientific << x_PALU << endl;
    //cout << "Solution using QR Decomposition:\n" << fixed << setprecision(5) << scientific << x_QR << endl;

    //Errore relativo
    double err_PALU = (x_PALU - c).norm() / c.norm();
    double err_QR = (x_QR - c).norm() / c.norm();

    cout << "Errore relativo con la decomoposzione PALU: " << fixed << setprecision(5) << scientific << err_PALU << endl;
    cout << "Errore relativo con la decomoposzione QR: " << fixed << setprecision(5) << scientific << err_QR << endl;
}

int main()
{
    // Sistema 1
    Matrix2d A1;
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01, -9.992887623566787e-01;
    Vector2d b1;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;
    Vector2d solution1;
    solution1 << -1, -1;

    cout << "Sistema 1" << endl;
    SystemSolution(A1, b1, solution1);

    // Sistema 2
    Matrix2d A2;
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;
    Vector2d b2;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;
    Vector2d solution2;
    solution2 << -1, -1;

    cout << "Sistema 2" << endl;
    SystemSolution(A2, b2, solution2);

    // Sistema 3
    Matrix2d A3;
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;
    Vector2d b3;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;
    Vector2d solution3;
    solution3 << -1, -1;

    cout << "Sistema 3" << endl;
    SystemSolution(A3, b3, solution3);

    return 0;
}
