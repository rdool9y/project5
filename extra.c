#include <emmintrin.h>
#include <omp.h>
#define blocksize 72
#define blocksize_Y 55

int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel, int kernel_x, int kernel_y)

{
int kern_cent_X = (kernel_x - 1)/2;
int kern_cent_Y = (kernel_y - 1)/2;

    __m128 kernel_vector;
    __m128 input_vector1, output_vector1, product_vector1, input_vector2, output_vector2, product_vector2, input_vector3, output_vector3, product_vector3;
    __m128 input_vector4, output_vector4, product_vector4, input_vector5, output_vector5, product_vector5, input_vector6, output_vector6, product_vector6;
    __m128 input_vector7, output_vector7, product_vector7, input_vector8, output_vector8, product_vector8, input_vector9, output_vector9, product_vector9;
    
    int padding_x = (kernel_x / 2); 
    int padding_y = (kernel_y / 2);
    int padded_size = (data_size_X + 2*padding_x) * (data_size_Y + 2*padding_y); 
    float* padded_in = malloc(padded_size * sizeof(float));

    int padded_row_length = data_size_X + 2*padding_x;
    
    
    int x,y,z;

    #pragma omp parallel for
        for(z = 0; z < padded_size; z++) {
         padded_in[z] = 0.0f;
        }
    #pragma omp parallel for private(x)
        for(y = 0; y < data_size_Y; y++) {
         for(x = 0; x < data_size_X; x++) {
                padded_in[(x+padding_x) + (y+padding_y)*(data_size_X + 2*padding_y)] = in[x+y*data_size_X];
         }
        }

    int a, b, i, j;
    float *input_base;
    float *output_base;
    int k;
    float local_kern[kernel_x*kernel_y];

    for(k = 0; k < kernel_x*kernel_y; k++) {
        local_kern[k] = kernel[k];
    }

    omp_set_num_threads(8);
    # pragma omp parallel shared(local_kern, padded_in, out, padded_row_length)
    {

// printf("There are %d threads running\n",omp_get_num_threads());

# pragma omp for private(a, b, i, j, x, y, kernel_vector, input_vector1, output_vector1, product_vector1, input_vector2, output_vector2, product_vector2, input_vector3, output_vector3,product_vector3,input_vector4,output_vector4,product_vector4,input_vector5,output_vector5,product_vector5,input_vector6,output_vector6,product_vector6,input_vector7,output_vector7,product_vector7,input_vector8,output_vector8,product_vector8,input_vector9,output_vector9,product_vector9,input_base,output_base) schedule(static,1)
        for(y = 0; y < data_size_Y; y+=blocksize_Y) {
          for(x = 0; x < data_size_X; x+=blocksize){
            for(a = x; a < x + blocksize && a <= data_size_X-36; a+=36) {
              for(b = y; b < y + blocksize_Y && b < data_size_Y; b++){
                    // set output vector to 0
                    output_vector1 = _mm_setzero_ps();
                    output_vector2 = _mm_setzero_ps();
                 output_vector3 = _mm_setzero_ps();
                 output_vector4 = _mm_setzero_ps();
                 output_vector5 = _mm_setzero_ps();
                 output_vector6 = _mm_setzero_ps();
                 output_vector7 = _mm_setzero_ps();
                 output_vector8 = _mm_setzero_ps();
                 output_vector9 = _mm_setzero_ps();
            
                    for(i = -kern_cent_X; i <= kern_cent_X; i++){ // inner loop; after all iterations, write 4 output sums
                        for(j = -kern_cent_Y; j <= kern_cent_Y; j++){


                            kernel_vector = _mm_load1_ps(local_kern + ((kern_cent_X-i) + (kern_cent_Y-j)*kernel_x));
                            input_base = padded_in + a + i + padding_x + (b+j+padding_y)*padded_row_length;
                        
                            input_vector1 = _mm_loadu_ps(input_base);
                            product_vector1 = _mm_mul_ps(kernel_vector, input_vector1);
                            output_vector1 = _mm_add_ps(output_vector1, product_vector1);
                
                            input_vector2 = _mm_loadu_ps(input_base + 4);
                            product_vector2 = _mm_mul_ps(kernel_vector, input_vector2);
                            output_vector2 = _mm_add_ps(output_vector2, product_vector2);

                         input_vector3 = _mm_loadu_ps(input_base + 8);
                            product_vector3 = _mm_mul_ps(kernel_vector, input_vector3);
                            output_vector3 = _mm_add_ps(output_vector3, product_vector3);

                         input_vector4 = _mm_loadu_ps(input_base + 12);
                            product_vector4 = _mm_mul_ps(kernel_vector, input_vector4);
                            output_vector4 = _mm_add_ps(output_vector4, product_vector4);

                         input_vector5 = _mm_loadu_ps(input_base + 16);
                            product_vector5 = _mm_mul_ps(kernel_vector, input_vector5);
                            output_vector5 = _mm_add_ps(output_vector5, product_vector5);

                         input_vector6 = _mm_loadu_ps(input_base + 20);
                            product_vector6 = _mm_mul_ps(kernel_vector, input_vector6);
                            output_vector6 = _mm_add_ps(output_vector6, product_vector6);

                         input_vector7 = _mm_loadu_ps(input_base + 24);
                            product_vector7 = _mm_mul_ps(kernel_vector, input_vector7);
                            output_vector7 = _mm_add_ps(output_vector7, product_vector7);

                         input_vector8 = _mm_loadu_ps(input_base + 28);
                            product_vector8 = _mm_mul_ps(kernel_vector, input_vector8);
                            output_vector8 = _mm_add_ps(output_vector8, product_vector8);

                         input_vector9 = _mm_loadu_ps(input_base + 32);
                            product_vector9 = _mm_mul_ps(kernel_vector, input_vector9);
                            output_vector9 = _mm_add_ps(output_vector9, product_vector9);
                        }
                    }
                    
                    output_base = out + a + b*data_size_X;
                
                    _mm_storeu_ps(output_base, output_vector1);
                    _mm_storeu_ps(output_base + 4, output_vector2);
                 _mm_storeu_ps(output_base + 8, output_vector3);
                 _mm_storeu_ps(output_base + 12, output_vector4);
                 _mm_storeu_ps(output_base + 16, output_vector5);
                 _mm_storeu_ps(output_base + 20, output_vector6);
                 _mm_storeu_ps(output_base + 24, output_vector7);
                 _mm_storeu_ps(output_base + 28, output_vector8);
                 _mm_storeu_ps(output_base + 32, output_vector9);

                }
            }
        }
     }
    

    } // end parallel
    //printf("done with parallel\n");
    float output_float, kernel_float, input_float, product_float;

    for(b = 0; b < data_size_Y; b++) {
        for(a = (data_size_X/36)*36; a <= data_size_X-4; a+=4) {
            
            output_vector1 = _mm_setzero_ps();
        
            for(i = -kern_cent_X; i <= kern_cent_X; i++){ // inner loop : all kernel elements
                for(j = -kern_cent_Y; j <= kern_cent_Y; j++){
                    kernel_vector = _mm_load1_ps(local_kern + ((kern_cent_X-i) + (kern_cent_Y-j)*kernel_x));
       
                    input_vector1 = _mm_loadu_ps(padded_in + a + i + padding_x + (b+j+padding_y)*padded_row_length);
                    product_vector1 = _mm_mul_ps(kernel_vector, input_vector1);
                    output_vector1 = _mm_add_ps(output_vector1, product_vector1);

                    _mm_storeu_ps(out + a + b*data_size_X, output_vector1);
                }
            }
            
        }
        for(; a < data_size_X; a++) {
            output_float = 0.0f;
            
            for(i = -kern_cent_X; i <= kern_cent_X; i++){ // inner loop : all kernel elements
                for(j = -kern_cent_Y; j <= kern_cent_Y; j++){
            
                    product_float = local_kern[(kern_cent_X - i) + (kern_cent_Y-j)*kernel_x] * padded_in[(a+i+padding_x)+(b+j+padding_y)*(data_size_X+2*padding_y)];
                    output_float += product_float;
                }
            }
            out[a + b*data_size_X] = output_float;
       
        }
    }
 
    free(padded_in);
    
    return 1;
}




