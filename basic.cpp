#include <iostream>
#include <vector>

// main function
int main(int argc, char *argv[])
{
    // check for correct arguments
    if (argc < 4)
    {
        std::cout << "Usage: <dimension> <array size> <number of queries>" << std::endl;
        return 1; // exit immediately
    }

    // command line input = <dimension> <array size> <number of queries>
    int dimension = std::stoi(argv[1]);
    int array_size = std::stoi(argv[2]);
    int query_max = std::stoi(argv[3]);

    int queries = 0; // how many queries have been done

    std::vector<float> prediction_vector(array_size, 0.0); // intialize prediction vector of size array_size and zeroes

    std::vector<bool> queried(array_size, false); // boolean vector to check if we've already query index

    float value; // TITAN's response value

    // loop to query TITAN
    for (int i = 0; i < array_size && queries < query_max; i++) // loop if we havent reached the end of array_size and max queries
    {
        // std::cout << "Querying Index: ";
        std::cout << i << std::endl
                  << std::flush; // output index for TITAN to read

        std::cin >> value;            // store TITAN input
        prediction_vector[i] = value; // put into prediction vector
        queried[i] = true;            // mark that spot as queried
        queries++;                    // increment number of queries made

        // query last index if next query is out of bounds
        if (i == array_size - 2 && queries < query_max)
        {
            // std::cout << "Querying Index: ";
            std::cout << array_size - 1 << std::endl << std::flush;

            std::cin >> value;
            prediction_vector[array_size - 1] = value;
            queried[array_size - 1] = true;
            queries++;
        }
    }

    // if we reach end of array and still have queries left, query leftover spots
    if (queries < query_max)
    {
        for (int j = 0; j < array_size && queries < query_max; j++)
        {
            // check if index has been queried already and increment if so
            if (queried[j] == false)
            {
                // output query
                // std::cout << "Querying Index: ";
                std::cout << j << std::endl
                          << std::flush;

                // read TITAN input
                std::cin >> value;
                prediction_vector[j] = value;
                queried[j] = true;
                queries++;
            }
        }
    }

    // print prediction values
    for (int k = 0; k < array_size; k++)
    {
        std::cout << prediction_vector[k];
        if (k < array_size - 1)
            std::cout << " ";
    }
    std::cout << std::endl;

    return 0;
}