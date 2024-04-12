#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <vector>

using namespace std;

template <typename T> void lu(const vector<vector<T>>& a, vector<vector<T>>& l, vector<vector<T>>& u, int n)
{
    int i = 0, j = 0, k = 0;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (j < i)
                l[j][i] = 0;
            else
            {
                l[j][i] = a[j][i];
                for (k = 0; k < i; k++)
                {
                    l[j][i] = l[j][i] - l[j][k] * u[k][i];
                }
            }
        }
        for (j = 0; j < n; j++)
        {
            if (j < i)
                u[i][j] = 0;
            else if (j == i)
                u[i][j] = 1;
            else
            {
                u[i][j] = a[i][j] / l[i][i];
                for (k = 0; k < i; k++)
                {
                    u[i][j] = u[i][j] - ((l[i][k] * u[k][j]) / l[i][i]);
                }
            }
        }
    }
}

template <typename T> void output(const vector<vector<T>>& x, int n)
{
    int i = 0, j = 0;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            cout << x[i][j] << "\t\t";
        }
        cout << "\n";
    }
}

template <typename T> void output2(const vector<T>& x, int n)
{
    int i = 0;
    for (i = 0; i < n; i++)
    {
        cout << x[i];

        cout << "\n";
    }
}

template <typename T> void LYCompute(const vector<vector<T>> L, const vector<T>& B, vector<T>& Y, int n)
{
    for (int i = 0; i < n; i++)
    {
        T sum = 0;
        T value = L[i][i];
        for (int j = 0; j < n; j++)
        {
            T temp = L[i][j] * Y[j];
            sum += temp;

        }
        Y[i] = (B[i] - sum) / value;
    }

}

template <typename T> void UXCompute(const vector<vector<T>>& U, const vector<T>& Y, vector<T>& X, int n)
{
    for (int i = n - 1; i >= 0; i--) {
        if (i == (n - 1))
            X[n - 1] = Y[n - 1];
        else {
            T value = Y[i];
            T temp = 0;
            for (int j = n - 1; j > 0; j--) {
                temp -= U[i][j] * X[j];
            }
            X[i] = temp + value;
        }
    }
}

template <typename T> void solve(const vector<vector<T>>& A, const vector<T>& B, vector<T>& X, int n) {
    vector<vector<T>> L(n, vector<T>(n, 0.0f));
    vector<vector<T>> U(n, vector<T>(n, 0.0f));
    vector<T> Y(n, 0.0f);
    // A = L * U
    lu(A, L, U, n);
    // L * Y = B
    LYCompute(L, B, Y, n);
    // U * X = Y
    UXCompute(U, Y, X, n);
}