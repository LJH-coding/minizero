file(GLOB_RECURSE SRCS *.cpp)

add_library(environment ${SRCS})
target_include_directories(
    environment PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    base
    atari
    connect6
    go
    gomoku
    hex
    killallgo
    nogo
    othello
    rubiks
    tictactoe
    stochastic
    stochastic/puzzle2048
    stochastic/tetris
)
target_link_libraries(
    environment
    config
    utils
    ale::ale-lib
    ${OpenCV_LIBS}
)
