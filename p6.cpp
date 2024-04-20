#include <iostream>
#include <opencv2/opencv.hpp>
#include <mpi.h>

using namespace cv;

const int IMAGE_WIDTH = 700;
const int IMAGE_HEIGHT = 700;
const int DEFAULT_MAX_ITERATIONS = 1000;

int Mandelbrot(double cr, double ci, int maxIterations) {
    double zr = 0;
    double zi = 0;
    int iterations = 0;
    while (zr * zr + zi * zi < 2 && iterations < maxIterations) {
        double zr_new = zr * zr - zi * zi + cr;
        double zi_new = 2 * zr * zi + ci;
        zr = zr_new;
        zi = zi_new;
        iterations++;
    }
    return iterations;
}

Vec3b getColor(int iterations) {
    if (iterations >= 255) {
        return Vec3b(0, 0, 0);
    }
    else {
        return Vec3b((iterations % 8 + 1) / 2.0 * 255, (iterations % 27 + 1) / 3.0 * 255, (iterations % 97 + 1) / 5.5 * 255);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, numProcesses;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    int maxIterations = DEFAULT_MAX_ITERATIONS;
    if (argc > 1) {
        maxIterations = atoi(argv[1]);
    }

    int rowsPerProcess = IMAGE_HEIGHT / numProcesses;
    int startRow = rank * rowsPerProcess;
    int endRow = (rank + 1) * rowsPerProcess;

    Mat fractalImage(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);

    for (int y = startRow; y < endRow; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            double cr = -2.0 + (1.0 + 2.0) * x / (IMAGE_WIDTH - 1);
            double ci = -1.5 + (1.5 + 1.5) * y / (IMAGE_HEIGHT - 1);

            int iterations = Mandelbrot(cr, ci, maxIterations);
            fractalImage.at<Vec3b>(y, x) = getColor(iterations);
        }
    }

    Mat resultImage;
    if (rank == 0) {
        resultImage = Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
    }

    MPI_Gather(fractalImage.data, IMAGE_WIDTH * rowsPerProcess * 3, MPI_BYTE,
        resultImage.data, IMAGE_WIDTH * rowsPerProcess * 3, MPI_BYTE, 0, MPI_COMM_WORLD);

    MPI_Finalize();

    if (rank == 0) {
        imshow("Total", resultImage);
        waitKey(0);
    }

    return 0;
}
