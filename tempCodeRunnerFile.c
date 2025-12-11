#include <stdio.h>

int main() {
    int R, C;
    scanf("%d", &R);
    scanf("%d", &C);
    int matrix[R][C];
    // Input matrix elements
    for(int i = 0; i < R; i++) {
        for(int j = 0; j < C; j++) {
            scanf("%d", &matrix[i][j]);
        }
    }
    // Find largest row sum
    int maxRowSum = 0;
    for(int i = 0; i < R; i++) {
        int rowSum = 0;
        for(int j = 0; j < C; j++) {
            rowSum += matrix[i][j];
        }
        if(rowSum > maxRowSum) {
            maxRowSum = rowSum;
        }
    }
    // Find largest column sum
    int maxColSum = 0;
    for(int j = 0; j < C; j++) {
        int colSum = 0;
        for(int i = 0; i < R; i++) {
            colSum += matrix[i][j];
        }
        if(colSum > maxColSum) {
            maxColSum = colSum;
        }
    }
    // Calculate and print lucky number
    int luckyNumber = maxRowSum + maxColSum;
    printf("%d\n", luckyNumber);

    return 0;
}
