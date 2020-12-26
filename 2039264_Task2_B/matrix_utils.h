
#ifndef MATRIX_UTILS_H_
#define MATRIX_UTILS_H_


float** random_matrix(const int n_rows, const int n_columns);

void print_matrix(float** matrix, int n_rows, int n_columns);

float** add(float** A, float** B, const int n_rows, int n_columns);


float** zero_matrix(const int n_rows, const int n_columns);

void delete_matrix(float** matrix, const int n_rows, int n_columns);


#endif /* MATRIX_UTILS_H_ */
