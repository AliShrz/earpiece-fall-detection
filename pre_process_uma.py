from scripts.uma import process_UMA


def main():      
    
   input_folder = r"DataSet/UMAFall_Dataset"

    # Path to the directory where we want to save the datasets
   output_folder = r"output_uma"
    

   process_UMA(input_folder, output_folder)
    
      
if __name__ == "__main__":
     main()