# splay_tree.py

class Node:
    """
    Represents a node in the splay tree.
    Each node has a key, an optional value, left and right children, and a parent.
    """
    def __init__(self, key, value=None):
        self.key = key
        self.value = value  # Store the embedding vector or any associated value
        self.left = None
        self.right = None
        self.parent = None

class SplayTree:
    """
    Traditional implementation of a splay tree.
    The tree supports insertion, search, and splaying operations to bring nodes closer to the root.
    """
    def __init__(self):
        self.root = None
        self.total_rotations = 0  # To track the number of rotations for performance metrics

    def _right_rotate(self, x):
        """Performs a right rotation around the given node x."""
        y = x.left
        if y is not None:
            x.left = y.right
            if y.right is not None:
                y.right.parent = x
            y.parent = x.parent

            if x.parent is None:  # x is root
                self.root = y
            elif x == x.parent.right:
                x.parent.right = y
            else:
                x.parent.left = y

            y.right = x
            x.parent = y
            self.total_rotations += 1

    def _left_rotate(self, x):
        """Performs a left rotation around the given node x."""
        y = x.right
        if y is not None:
            x.right = y.left
            if y.left is not None:
                y.left.parent = x
            y.parent = x.parent

            if x.parent is None:  # x is root
                self.root = y
            elif x == x.parent.left:
                x.parent.left = y
            else:
                x.parent.right = y

            y.left = x
            x.parent = y
            self.total_rotations += 1

    def _splay(self, x):
        """Splays the given node x to the root of the tree."""
        if x is None:
            return
        while x.parent is not None:
            if x.parent.parent is None:
                # Zig step
                if x == x.parent.left:
                    self._right_rotate(x.parent)
                else:
                    self._left_rotate(x.parent)
            else:
                # Zig-Zig or Zig-Zag step
                p = x.parent
                g = p.parent
                if x == p.left and p == g.left:
                    # Zig-Zig step (left-left)
                    self._right_rotate(g)
                    self._right_rotate(p)
                elif x == p.right and p == g.right:
                    # Zig-Zig step (right-right)
                    self._left_rotate(g)
                    self._left_rotate(p)
                elif x == p.right and p == g.left:
                    # Zig-Zag step (left-right)
                    self._left_rotate(p)
                    self._right_rotate(g)
                else:
                    # Zig-Zag step (right-left)
                    self._right_rotate(p)
                    self._left_rotate(g)

    def insert(self, key, value=None):
        """Inserts a key-value pair into the splay tree and splays the newly inserted node to the root."""
        z = self.root
        p = None

        while z is not None:
            p = z
            if key < z.key:
                z = z.left
            elif key > z.key:
                z = z.right
            else:
                # If the key already exists, update its value and splay it
                z.value = value
                self._splay(z)
                return

        z = Node(key, value)
        z.parent = p

        if p is None:  # The tree was empty
            self.root = z
        elif key < p.key:
            p.left = z
        else:
            p.right = z

        self._splay(z)

    def search(self, key):
        """Searches for a key in the splay tree and splays it to the root if found."""
        z = self.root
        while z is not None:
            if key == z.key:
                self._splay(z)
                return z.value  # Return the associated value
            elif key < z.key:
                z = z.left
            else:
                z = z.right
        return None  # Key not found

    def _find_node(self, key):
        """Finds a node with the given key without splaying it."""
        z = self.root
        while z is not None:
            if key == z.key:
                return z
            elif key < z.key:
                z = z.left
            else:
                z = z.right
        return None

    def access(self, key):
        """
        Accesses a node with the given key, performing a splay operation.
        This method provides compatibility with the MCSTSQSplayTree interface.
        """
        # If the key doesn't exist in the tree, insert it
        node = self._find_node(key)
        if node is None:
            self.insert(key)
            return self.root
        else:
            self._splay(node)
            return node

    def delete(self, key):
        """Deletes a node with the given key from the splay tree."""
        node = self.search(key)
        if node is None:
            return  # Key not found, nothing to delete

        self._splay(node)

        if node.left is None:
            self._transplant(node, node.right)
        elif node.right is None:
            self._transplant(node, node.left)
        else:
            y = self._subtree_minimum(node.right)
            if y.parent != node:
                self._transplant(y, y.right)
                y.right = node.right
                y.right.parent = y
            self._transplant(node, y)
            y.left = node.left
            y.left.parent = y

    def _transplant(self, u, v):
        """Replaces the subtree rooted at node u with the subtree rooted at node v."""
        if u.parent is None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        if v is not None:
            v.parent = u.parent

    def _subtree_minimum(self, node):
        """Returns the node with the minimum key in the subtree rooted at the given node."""
        while node.left is not None:
            node = node.left
        return node

    def _get_depth(self, node):
        """Calculates the depth of a node from the root."""
        depth = 0
        current = node
        while current != self.root and isinstance(current, Node):
            current = current.parent
            depth += 1
        return depth

    def _record_rotation(self):
        """Increment rotation count for rotation tracking."""
        self.total_rotations += 1
