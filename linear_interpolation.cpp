#include "linear_interpolation.h"

// Linear interpolation function implementation
void linear_interpolate(std::vector<float> &prediction_vector, std::vector<bool> &queried)
{
    int array_size = prediction_vector.size();

    for (int i = 0; i < array_size; i++) // loop through entire array
    {
        if (!queried[i]) // if index isnt queried
        {
            // find nearest left value
            int left = i - 1;
            while (left >= 0 && !queried[left])
                left--;

            // find nearest right value
            int right = i + 1;
            while (right < array_size && !queried[right])
                right++;

            float left_val = (left >= 0) ? prediction_vector[left] : 0.0f;
            float right_val = (right < array_size) ? prediction_vector[right] : left_val;

            // linear interpolation formula
            if (right != left)
            {
                prediction_vector[i] = left_val + (right_val - left_val) * (float)(i - left) / (float)(right - left);
            }
            else
            {
                prediction_vector[i] = left_val; // fallback if no right value
            }

            queried[i] = true; // mark as filled
        }
    }
}
