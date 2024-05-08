# Homework
Public repository and stub/testing code for Homework 0 of 10-714.

---

## My solution

### Question 1

Just simply return `x+y`.

### Question 2

Take the parsing process of image file as an example.

Firstly, `struct.uppack` was utilized to parse the file because the first 4 byte in image file represent a magic number, the number of images, the number of rows per image, the number of columns per image, respectively.

Secondly, `np.frombuffer` was employed to read data about image. To obtain the correct format, we first apply reshape method to get a NumPy array with shape(60000 x 784).  Then we use the astype method to transform data-type from unsigned int to float32.

The same for parsing label file.

### Q3-Q6

It has been solved, and I will explain the specific approach afterward.
