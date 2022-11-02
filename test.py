import pandas
import numpy as np

num_rows = 7;
df2 = pandas.DataFrame(
                        {
                        'a' : ['one', 'one', 'two', 'three', 'two', 'one', 'six'],
                        'b' : ['x', 'y', 'y', 'x', 'y', 'x', 'x'],
                        'c' : np.random.randn(num_rows)
                        }
                      )
print(df2)

a_attribute_list = ['one', 'two', 'three', 'six']; #Or use list(set(df2['a'].values)), but that doesn't guarantee ordering.
b_attribute_list = ['x','y']

a_membership = [ np.reshape(np.array(df2['a'].values == elem).astype(np.float64),   (num_rows,1)) for elem in a_attribute_list ]
b_membership = [ np.reshape((df2['b'].values == elem).astype(np.float64), (num_rows,1)) for elem in b_attribute_list ]
c_column =  np.reshape(df2['c'].values, (num_rows,1))


design_matrix_a = np.hstack(tuple(a_membership))
design_matrix_b = np.hstack(tuple(b_membership))
design_matrix = np.hstack(( design_matrix_a, design_matrix_b, c_column ))

# Print out the design matrix to see that it's what you want.
for row in design_matrix:
    print (row)