template <typename T>
bool rms_norm(T *__restrict__ output, float *__restrict__ rms_output,
              const T *__restrict__ input, const T *__restrict__ weight,
              int rows, int columns, float epsilon, cudaStream_t stream);
