class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        return f"ListNode(val={self.val})"


def reverse_linked_list(head: ListNode | None) -> ListNode | None:
    """Reverse a singly linked list iteratively."""
    prev = None
    curr = head

    while curr is not None:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt

    return prev


def build_linked_list(values: list[int]) -> ListNode | None:
    head = None
    tail = None
    for v in values:
        node = ListNode(v)
        if head is None:
            head = tail = node
        else:
            tail.next = node
            tail = node
    return head


def to_list(head: ListNode | None) -> list[int]:
    out = []
    curr = head
    while curr is not None:
        out.append(curr.val)
        curr = curr.next
    return out


if __name__ == "__main__":
    head = build_linked_list([1, 2, 3, 4, 5])
    new_head = reverse_linked_list(head)
    print(to_list(new_head))  # [5, 4, 3, 2, 1]
