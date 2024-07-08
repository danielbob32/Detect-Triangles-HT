# Hough Transform Triangle Detection

This project involves the detection of triangles in images using the Hough Transform technique. It processes an image to detect edges, applies the Hough Transform to identify lines, and then evaluates the intersections of these lines to determine the presence and type of triangles.
This project was developed collaboratively by Daniel Bobritski and Alex Chudnovsky.

### Project Structure

- **CV_ex1.ipynb**: The main notebook to execute the triangle detection.
- **requirements.txt**: A file listing the dependencies required to run the project.
- **assets/**: A directory containing images and other assets used in the project.

### Key Features

- ![image](https://github.com/danielbob32/Computer-Vision---Identify-Triangles-using-Hough-Transform/assets/120675110/c5dfa5bc-62c0-463c-a8be-7b5f63dff6f6)
 **Edge Detection**: Uses Canny edge detection to find edges in the image.
- ![image](https://github.com/danielbob32/Computer-Vision---Identify-Triangles-using-Hough-Transform/assets/120675110/ea733037-a2e9-44a6-b00b-77f4e6b8c616)
 **Hough Transform**: Applies the Hough Transform to detect lines from the edges.
- ![image](https://github.com/danielbob32/Computer-Vision---Identify-Triangles-using-Hough-Transform/assets/120675110/db0a0158-9319-4428-9554-e31136d8ec6f)
 **Triangle Classification**: Identifies and classifies triangles based on the intersections of the detected lines.

### Dependencies

To run the project, install the required packages using the following command:

```bash
pip install -r requirements.txt
```

### Example Usage

Open and run the CV_ex1.ipynb notebook in Jupyter to see the triangle detection in action.
