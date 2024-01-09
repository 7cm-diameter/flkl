if __name__ == "__main__":
    from pyno.com import check_connected_board_info

    boards = check_connected_board_info()

    for board in boards:
        print(f"{board.board} (serial number: {board.serial_number}) at {board.port}")
