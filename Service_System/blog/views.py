from datetime import timedelta

from django.shortcuts import render, get_object_or_404, redirect
from django.utils import timezone

from rest_framework import viewsets, filters, status
from rest_framework.response import Response

from .models import Post
from .forms import PostForm
from .serializers import PostSerializer


# =========================================================
# 일반 웹 페이지
# =========================================================


def post_list(request):
    """
    게시된 이미지 목록을 출력한다.

    YOLO 검출 결과를 빠르게 확인할 수 있도록
    최근 게시물부터 출력한다.
    """

    posts = (
        Post.objects
        .filter(
            published_date__lte=timezone.now()
        )
        .order_by(
            '-published_date'
        )
    )

    return render(
        request,
        'blog/post_list.html',
        {
            'posts': posts
        }
    )


def post_detail(request, pk):
    """
    하나의 게시물을 상세 조회한다.
    """

    post = get_object_or_404(
        Post,
        pk=pk
    )

    return render(
        request,
        'blog/post_detail.html',
        {
            'post': post
        }
    )


def post_new(request):
    """
    웹 브라우저에서 새로운 게시물을 생성한다.
    """

    if request.method == "POST":

        form = PostForm(
            request.POST,
            request.FILES
        )

        if form.is_valid():

            post = form.save(
                commit=False
            )

            post.author = request.user

            post.published_date = (
                timezone.now()
            )

            post.save()

            return redirect(
                'post_detail',
                pk=post.pk
            )

    else:

        form = PostForm()

    return render(
        request,
        'blog/post_edit.html',
        {
            'form': form
        }
    )


def post_edit(request, pk):
    """
    기존 게시물을 수정한다.
    """

    post = get_object_or_404(
        Post,
        pk=pk
    )

    if request.method == "POST":

        form = PostForm(
            request.POST,
            request.FILES,
            instance=post
        )

        if form.is_valid():

            post = form.save(
                commit=False
            )

            post.author = (
                request.user
            )

            post.published_date = (
                timezone.now()
            )

            post.save()

            return redirect(
                'post_detail',
                pk=post.pk
            )

    else:

        form = PostForm(
            instance=post
        )

    return render(
        request,
        'blog/post_edit.html',
        {
            'form': form
        }
    )


def js_test(request):
    """
    JavaScript 테스트 페이지.
    """

    return render(
        request,
        'blog/js_test.html',
        {}
    )


# =========================================================
# REST API
# =========================================================


class BlogImages(
    viewsets.ModelViewSet
):
    """
    이미지 게시물 REST API.

    YOLO Edge Client 및 Android Client에서 사용한다.

    주요 기능
    1. 게시물 조회 / 생성 / 수정 / 삭제
    2. YOLO 검출 이벤트 검색 및 정렬
    3. 반복 검출 이벤트 중복 등록 방지
    """

    queryset = (
        Post.objects
        .all()
        .order_by(
            '-created_date'
        )
    )

    serializer_class = (
        PostSerializer
    )


    # =====================================================
    # 편의 기능 1
    #
    # YOLO 검출 이벤트 검색 및 정렬
    #
    # 예:
    #
    # /api_root/Post/?search=person
    #
    # /api_root/Post/?ordering=-created_date
    #
    # /api_root/Post/?search=person&ordering=-created_date
    #
    # =====================================================

    filter_backends = [
        filters.SearchFilter,
        filters.OrderingFilter,
    ]


    # -----------------------------------------------------
    # 검색 대상
    #
    # YOLO에서는 검출 객체 이름이
    # title / text에 저장된다.
    #
    # 따라서 person, car 등의 객체명을 검색할 수 있다.
    # -----------------------------------------------------

    search_fields = [
        'title',
        'text',
    ]


    # -----------------------------------------------------
    # 정렬 가능 필드
    # -----------------------------------------------------

    ordering_fields = [
        'created_date',
        'published_date',
        'title',
    ]


    # -----------------------------------------------------
    # 기본 정렬
    #
    # 가장 최근 검출 이벤트부터 출력
    # -----------------------------------------------------

    ordering = [
        '-created_date'
    ]


    # =====================================================
    # 편의 기능 2
    #
    # 반복 검출 이벤트 중복 등록 방지
    #
    # 최근 10초 이내에
    #
    # author
    # title
    # text
    #
    # 값이 동일한 게시물이 존재하면
    # 새로운 게시물을 생성하지 않는다.
    #
    # YOLO 측 ChangeDetection의 0 -> 1 감지와
    # Server 측 중복 검사를 함께 사용한다.
    # =====================================================

    def create(
        self,
        request,
        *args,
        **kwargs
    ):

        # -------------------------------------------------
        # POST 요청 데이터
        # -------------------------------------------------

        author_id = request.data.get(
            'author'
        )

        title = request.data.get(
            'title',
            ''
        )

        text = request.data.get(
            'text',
            ''
        )


        # None 값에 대한 안전 처리
        if title is None:
            title = ''

        if text is None:
            text = ''


        title = str(
            title
        ).strip()

        text = str(
            text
        ).strip()


        # -------------------------------------------------
        # 중복 검사를 수행할 수 있는 경우
        #
        # author / title 값이 존재할 때 검사
        # -------------------------------------------------

        if (
            author_id
            and title
        ):

            duplicate_since = (
                timezone.now()
                - timedelta(
                    seconds=10
                )
            )


            duplicate_post = (
                Post.objects
                .filter(
                    author_id=author_id,
                    title=title,
                    text=text,
                    created_date__gte=duplicate_since
                )
                .order_by(
                    '-created_date'
                )
                .first()
            )


            # ---------------------------------------------
            # 동일 게시물이 최근 10초 내 존재
            # ---------------------------------------------

            if (
                duplicate_post
                is not None
            ):

                return Response(
                    {
                        'detail':
                            'Duplicate post prevented.',

                        'message':
                            '동일한 검출 이벤트가 최근에 이미 등록되었습니다.',

                        'post_id':
                            duplicate_post.pk,

                        'duplicate_window_seconds':
                            10,
                    },
                    status=(
                        status.HTTP_409_CONFLICT
                    )
                )


        # -------------------------------------------------
        # 중복이 아니면 기존 DRF 생성 로직 실행
        # -------------------------------------------------

        return super().create(
            request,
            *args,
            **kwargs
        )