/*
 * matrix_utils.h
 *
 *  Created on: 12-Jan-2020
 */

#ifndef MATRIX_UTILS_H_
#define MATRIX_UTILS_H_


void print_matrix(float** matrix, int n_rows, int n_columns);

float** random_matrix(const int n_rows, const int n_columns);

float** zero_matrix(const int n_rows, const int n_columns);

void remove_matrix(float** matrix, const int n_rows, int n_columns);

float** add(float** A, float** B, const int n_rows, int n_columns);

#endif /* MATRIX_UTILS_H_ */
