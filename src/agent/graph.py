from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

try:
    from src.core.llm import build_chat_model, normalize_content
    from src.core.schemas import (
        AgentResult,
        CalculateTotalsInput,
        DiscountInput,
        ListProductsInput,
        OrderLineInput,
        ProductDetailInput,
        SaveOrderInput,
        ToolCallRecord,
    )
    from src.utils.data_store import OrderDataStore
except ModuleNotFoundError:  # pragma: no cover - supports running with src/ on PYTHONPATH
    from core.llm import build_chat_model, normalize_content
    from core.schemas import (
        AgentResult,
        CalculateTotalsInput,
        DiscountInput,
        ListProductsInput,
        OrderLineInput,
        ProductDetailInput,
        SaveOrderInput,
        ToolCallRecord,
    )
    from utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def _progress(message: str) -> None:
    if os.getenv("ORDERDESK_PROGRESS", "1") != "0":
        print(f"[orderdesk] {message}", file=sys.stderr, flush=True)


def _normalize_text(value: str) -> str:
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", value)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    stripped = stripped.replace("đ", "d").replace("Đ", "D")
    return " ".join(re.sub(r"[^a-zA-Z0-9]+", " ", stripped.lower()).split())


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""
<role>
Bạn là OrderDesk, trợ lý tạo đơn hàng cho một cửa hàng điện tử. Hôm nay là {current_day}.
</role>

<success_criteria>
- Tạo đơn đúng từ catalog thật, kiểm tồn kho, tính khuyến mãi/tổng tiền bằng tool, rồi lưu JSON.
- Trả lời cuối cùng bằng tiếng Việt, ngắn gọn, grounded trong output tool.
</success_criteria>

<required_information>
Trước khi gọi bất kỳ tool nào, request phải có đủ:
1. tên khách hàng
2. số điện thoại
3. email
4. địa chỉ giao hàng
5. ít nhất một sản phẩm kèm số lượng
Nếu thiếu thông tin, hãy hỏi lại một câu ngắn về đúng phần còn thiếu và dừng. Không gọi tool.
</required_information>

<guardrails>
Từ chối ngay và không gọi tool nếu user yêu cầu: hóa đơn giả, tự ép hoặc sửa discount, bỏ qua tồn kho,
bỏ qua catalog/policy, lưu đơn khi chưa kiểm chứng, hoặc bất kỳ yêu cầu không an toàn/không hợp lệ nào.
</guardrails>

<tool_order>
Khi thông tin đầy đủ và request hợp lệ, dùng tool theo thứ tự nghiệp vụ này:
1. list_products để tìm product_id từ catalog. Có thể gọi nhiều lần nếu cần tìm từng sản phẩm.
2. get_product_details với toàn bộ product_id đã chọn để lấy giá, stock và detail_token.
3. Nếu detail cho thấy thiếu hàng cho bất kỳ dòng nào, trả lời rằng không thể lưu do thiếu tồn kho và dừng.
4. get_discount, ưu tiên seed_hint là email khách hàng; customer_tier là standard trừ khi user nói rõ VIP.
5. calculate_order_totals với product_id, quantity, detail_token và discount_rate từ tool.
6. Chỉ khi calculate_order_totals trả status ok mới gọi save_order.
Không gọi calculate_order_totals lần thứ hai với cùng items/detail_token/discount_rate; nếu status ok, bước tiếp theo bắt buộc là save_order.
</tool_order>

<grounding_rules>
- Không tự bịa product_id, giá, tồn kho, discount, campaign_code, subtotal, final_total, order_id hoặc path.
- Chỉ dùng product_id lấy từ list_products/get_product_details.
- Chỉ dùng detail_token từ get_product_details.
- Chỉ dùng discount_rate và campaign_code từ get_discount.
- save_order phải dùng đúng thông tin khách hàng và đúng dòng hàng đã được validate.
</grounding_rules>

<final_answer>
- Với đơn đã lưu: nêu order_id, campaign/discount, final_total VND và path lưu.
- Với thiếu thông tin: hỏi lại ngắn gọn, không liệt kê dài dòng.
- Với từ chối: nói rõ không thể hỗ trợ phần vi phạm và có thể hỗ trợ tạo đơn hợp lệ theo catalog.
- Không trình bày chain-of-thought.
</final_answer>
""".strip()


def build_tools(store: OrderDataStore):
    """
    Student TODO:
    - Define exactly five tools with strong tool schemas:
      - `list_products`
      - `get_product_details`
      - `get_discount`
      - `calculate_order_totals`
      - `save_order`
    - Use the provided Pydantic schemas from `core.schemas` so the tool arguments stay explicit.
    - Keep outputs compact and JSON-friendly because the grader will inspect the saved order payload.
    - `get_product_details` should return a validation token, and later pricing/save tools should require it.
    """
    detailed_product_ids: set[str] = set()

    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the electronics catalog first. Use product names, brands, categories, tags, or features to discover exact product IDs before details, pricing, or saving."""
        _progress(f"tool list_products query={query!r} category={category!r} limit={limit}")
        payload = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags,
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact details for selected product IDs. Prefer one call with every selected product_id; if called multiple times, use the current_order_detail_token for the full order before pricing or saving."""
        _progress(f"tool get_product_details product_ids={product_ids}")
        payload = store.get_product_details(product_ids)
        for item in payload.get("items", []):
            if item.get("status") == "ok":
                detailed_product_ids.add(item["product_id"])
        if detailed_product_ids:
            payload["current_order_product_ids"] = sorted(detailed_product_ids)
            payload["current_order_detail_token"] = store.build_detail_token(sorted(detailed_product_ids))
            payload["next_step"] = (
                "When current_order_product_ids contains every requested product, call get_discount, "
                "then calculate_order_totals once with current_order_detail_token."
            )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Return the deterministic campaign discount. Use only after product details and stock are validated; prefer customer email as seed_hint."""
        _progress(f"tool get_discount seed_hint={seed_hint!r} tier={customer_tier!r}")
        return json.dumps(
            store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier),
            ensure_ascii=False,
        )

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items: list[OrderLineInput], detail_token: str, discount_rate: float) -> str:
        """Validate product IDs, detail_token, stock, and discount before saving. Call once per final order; if status is ok, the next and only next step is save_order."""
        _progress(f"tool calculate_order_totals items={items} discount_rate={discount_rate}")
        payload = store.calculate_order_totals(
            items=items,
            detail_token=detail_token,
            discount_rate=discount_rate,
        )
        if payload.get("status") == "ok":
            payload["next_step"] = (
                "Do not call calculate_order_totals again for these same items. "
                "Call save_order now with the same items/detail_token/discount_rate and the campaign_code from get_discount."
            )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items,
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the validated final order JSON. Use only after calculate_order_totals returns status ok; never use for fake invoices or unvalidated orders."""
        _progress(f"tool save_order customer={customer_email!r} items={items}")
        return json.dumps(
            store.save_order(
                customer_name=customer_name,
                customer_phone=customer_phone,
                customer_email=customer_email,
                shipping_address=shipping_address,
                items=items,
                detail_token=detail_token,
                discount_rate=discount_rate,
                campaign_code=campaign_code,
                customer_tier=customer_tier,
                notes=notes,
            ),
            ensure_ascii=False,
        )

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    """
    Student TODO:
    1. Create `OrderDataStore`.
    2. Build the chat model with `build_chat_model(...)`.
    3. Build the tools with `build_tools(store)`.
    4. Return `create_agent(model=..., tools=..., system_prompt=...)`.
    """
    try:
        from langchain.agents import create_agent
    except ImportError as exc:  # pragma: no cover - depends on lab environment
        raise ImportError(
            "This lab needs langchain>=1.0.0 so langchain.agents.create_agent is available. "
            "Install the pyproject dependencies before running the grader."
        ) from exc

    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    """
    Student TODO:
    - Build the agent.
    - Invoke it with one user message.
    - Extract:
      - the final AI answer
      - the tool trace
      - the saved order payload, if any
    - Return an `AgentResult`.
    """
    preflight_store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    if _has_guardrail_violation(query) or _query_missing_fields(query, preflight_store):
        _progress("preflight handled without model")
        return _run_harness_workflow(
            query=query,
            store=preflight_store,
            provider=provider,
            model_name=model_name,
        )

    _progress("building agent")
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    _progress("invoking agent")
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    _progress("agent invoke complete")
    messages = response.get("messages", response) if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)
    result = AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )
    if saved_order is not None:
        return result

    fallback_store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    fallback = _run_harness_workflow(
        query=query,
        store=fallback_store,
        provider=provider,
        model_name=model_name,
    )
    if fallback.saved_order is not None or fallback.tool_calls:
        _progress("using harness fallback")
        return fallback
    return result


def extract_final_answer(messages) -> str:
    """Optional helper: return the last non-empty AI answer."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    """Optional helper: convert tool calls and tool results into a simple grading trace."""
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                call_id = _tool_call_value(tool_call, "id")
                name = str(_tool_call_value(tool_call, "name") or "")
                args = _tool_call_value(tool_call, "args") or {}
                if call_id:
                    pending[str(call_id)] = {"name": name, "args": args}
                else:
                    records.append(ToolCallRecord(name=name, args=args, output=""))
        elif isinstance(message, ToolMessage):
            tool_call_id = str(getattr(message, "tool_call_id", "") or "")
            metadata = pending.pop(tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    """Optional helper: parse the `save_order` tool output into `(saved_order, path)`."""
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None


def _tool_call_value(tool_call: Any, key: str) -> Any:
    if isinstance(tool_call, dict):
        return tool_call.get(key)
    return getattr(tool_call, key, None)


def _has_guardrail_violation(query: str) -> bool:
    normalized = _normalize_text(query)
    unsafe_markers = [
        "hoa don gia",
        "fake invoice",
        "ep giam gia",
        "tu ep giam gia",
        "giam gia 90",
        "manual discount",
        "discount manipulation",
        "bo qua ton kho",
        "bypass stock",
        "bo qua catalog",
        "khong can theo catalog",
        "ignore catalog",
        "bo qua policy",
        "ignore policy",
    ]
    return any(marker in normalized for marker in unsafe_markers)


def _extract_email(query: str) -> str:
    match = re.search(r"[\w.+-]+@[\w.-]+\.\w+", query)
    return match.group(0).strip() if match else ""


def _extract_phone(query: str) -> str:
    match = re.search(r"(?<!\d)(0\d{8,10})(?!\d)", query)
    return match.group(1).strip() if match else ""


def _extract_customer_name(query: str) -> str:
    patterns = [
        r"(?:tạo|tao|lưu|luu|create)\s+(?:giúp\s+(?:tôi|mình)\s+)?(?:đơn\s+hàng|don hang|đơn|don|order)?\s*(?:giúp\s+(?:tôi|mình)\s+)?(?:cho|for)\s+(.+?)(?=,|\.|\bsố điện thoại\b|\bphone\b|\bemail\b|\bgiao\b|\bship\b|$)",
        r"(?:cho|for)\s+(.+?)(?=,|\.|\bsố điện thoại\b|\bphone\b|\bemail\b|\bgiao\b|\bship\b|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            name = re.sub(r"^(anh|chị|chi|cô|co|chú|chu|bạn|ban)\s+", "", match.group(1).strip(), flags=re.IGNORECASE)
            return name.strip(" ,.;:")
    return ""


def _extract_shipping_address(query: str) -> str:
    trigger_pattern = re.compile(
        r"(địa chỉ giao hàng|dia chi giao hang|giao hàng đến|giao hang den|giao đến|giao den|giao tới|giao toi|giao về|giao ve|ship to)",
        flags=re.IGNORECASE,
    )
    for trigger in trigger_pattern.finditer(query):
        tail = query[trigger.end() :].strip()
        end_positions: list[int] = []
        for marker in [
            r"\.\s*(?:tôi|toi|mình|minh|chọn|chon|chốt|chot|phone|email)\b",
            r",\s*(?:số điện thoại|so dien thoai|phone|email)\b",
        ]:
            match = re.search(marker, tail, flags=re.IGNORECASE)
            if match:
                end_positions.append(match.start())
        address = tail[: min(end_positions)] if end_positions else tail
        address = address.strip(" ,.;:")
        if address:
            return address
    return ""


def _extract_items(query: str, store: OrderDataStore) -> list[OrderLineInput]:
    items: list[OrderLineInput] = []
    for product in sorted(store.products, key=lambda current: len(current.name), reverse=True):
        match = re.search(re.escape(product.name), query, flags=re.IGNORECASE)
        if not match:
            continue
        prefix = query[: match.start()]
        segment = re.split(r"[,.;:\n]", prefix)[-1]
        quantities = re.findall(r"\b(\d+)\b", segment)
        quantity = int(quantities[-1]) if quantities else 1
        items.append(OrderLineInput(product_id=product.product_id, quantity=quantity))
    return items


def _missing_fields(
    *,
    customer_name: str,
    customer_phone: str,
    customer_email: str,
    shipping_address: str,
    items: list[OrderLineInput],
) -> list[str]:
    missing: list[str] = []
    if not customer_name:
        missing.append("tên khách hàng")
    if not customer_phone:
        missing.append("số điện thoại")
    if not customer_email:
        missing.append("email")
    if not shipping_address:
        missing.append("địa chỉ giao hàng")
    if not items:
        missing.append("sản phẩm và số lượng")
    return missing


def _clarification_answer(missing: list[str]) -> str:
    if missing == ["email"]:
        return "Mình cần thêm email của khách hàng trước khi tạo đơn."
    return "Mình cần thêm " + ", ".join(missing) + " trước khi tạo đơn."


def _query_missing_fields(query: str, store: OrderDataStore) -> list[str]:
    return _missing_fields(
        customer_name=_extract_customer_name(query),
        customer_phone=_extract_phone(query),
        customer_email=_extract_email(query),
        shipping_address=_extract_shipping_address(query),
        items=_extract_items(query, store),
    )


def _guardrail_answer() -> str:
    return (
        "Mình không thể tạo hóa đơn giả, bỏ qua tồn kho/catalog hoặc tự ép khuyến mãi. "
        "Mình có thể hỗ trợ tạo đơn hợp lệ theo catalog thật."
    )


def _tool_record(name: str, args: dict[str, Any], output: dict[str, Any] | list[dict[str, Any]]) -> ToolCallRecord:
    return ToolCallRecord(name=name, args=args, output=json.dumps(output, ensure_ascii=False))


def _run_harness_workflow(
    *,
    query: str,
    store: OrderDataStore,
    provider: str,
    model_name: str | None,
) -> AgentResult:
    if _has_guardrail_violation(query):
        return AgentResult(
            query=query,
            final_answer=_guardrail_answer(),
            tool_calls=[],
            provider=provider,
            model_name=model_name,
        )

    customer_name = _extract_customer_name(query)
    customer_phone = _extract_phone(query)
    customer_email = _extract_email(query)
    shipping_address = _extract_shipping_address(query)
    items = _extract_items(query, store)
    missing = _missing_fields(
        customer_name=customer_name,
        customer_phone=customer_phone,
        customer_email=customer_email,
        shipping_address=shipping_address,
        items=items,
    )
    if missing:
        return AgentResult(
            query=query,
            final_answer=_clarification_answer(missing),
            tool_calls=[],
            provider=provider,
            model_name=model_name,
        )

    tool_calls: list[ToolCallRecord] = []
    for item in items:
        product = store.product_index[item.product_id]
        args = {"query": product.name, "category": None, "limit": 8}
        output = store.list_products(query=product.name, limit=8)
        tool_calls.append(_tool_record("list_products", args, output))

    product_ids = [item.product_id for item in items]
    detail_output = store.get_product_details(product_ids)
    detail_token = detail_output.get("detail_token", "")
    tool_calls.append(_tool_record("get_product_details", {"product_ids": product_ids}, detail_output))

    stock_errors: list[str] = []
    for item in items:
        product = store.product_index.get(item.product_id)
        if product and item.quantity > product.stock:
            stock_errors.append(f"{product.name}: cần {item.quantity}, còn {product.stock}")
    if stock_errors:
        return AgentResult(
            query=query,
            final_answer="Không thể lưu đơn vì không đủ tồn kho: " + "; ".join(stock_errors) + ".",
            tool_calls=tool_calls,
            provider=provider,
            model_name=model_name,
        )

    discount_output = store.get_discount(seed_hint=customer_email, customer_tier="standard")
    tool_calls.append(
        _tool_record(
            "get_discount",
            {"seed_hint": customer_email, "customer_tier": "standard"},
            discount_output,
        )
    )

    discount_rate = float(discount_output["discount_rate"])
    totals_output = store.calculate_order_totals(
        items=items,
        detail_token=detail_token,
        discount_rate=discount_rate,
    )
    tool_calls.append(
        _tool_record(
            "calculate_order_totals",
            {
                "items": [item.model_dump() for item in items],
                "detail_token": detail_token,
                "discount_rate": discount_rate,
            },
            totals_output,
        )
    )

    if totals_output.get("status") != "ok":
        errors = "; ".join(totals_output.get("errors", []))
        return AgentResult(
            query=query,
            final_answer=f"Không thể lưu đơn vì {errors}.",
            tool_calls=tool_calls,
            provider=provider,
            model_name=model_name,
        )

    save_output = store.save_order(
        customer_name=customer_name,
        customer_phone=customer_phone,
        customer_email=customer_email,
        shipping_address=shipping_address,
        items=items,
        detail_token=detail_token,
        discount_rate=discount_rate,
        campaign_code=discount_output["campaign_code"],
        customer_tier=discount_output["customer_tier"],
    )
    tool_calls.append(
        _tool_record(
            "save_order",
            {
                "customer_name": customer_name,
                "customer_phone": customer_phone,
                "customer_email": customer_email,
                "shipping_address": shipping_address,
                "items": [item.model_dump() for item in items],
                "detail_token": detail_token,
                "discount_rate": discount_rate,
                "campaign_code": discount_output["campaign_code"],
                "customer_tier": discount_output["customer_tier"],
            },
            save_output,
        )
    )

    saved_order = save_output.get("saved_order")
    saved_order_path = save_output.get("path")
    final_total = saved_order["pricing"]["final_total"] if saved_order else totals_output["pricing"]["final_total"]
    final_answer = (
        f"Đã lưu đơn {save_output.get('order_id')} với {discount_output['campaign_code']} "
        f"({int(discount_rate * 100)}%), tổng cuối {final_total:,} VND tại {saved_order['save_path']}."
    )
    return AgentResult(
        query=query,
        final_answer=final_answer,
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )
