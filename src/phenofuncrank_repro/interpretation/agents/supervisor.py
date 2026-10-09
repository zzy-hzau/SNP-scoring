"Supervisor Agent — the sole top-level controller."

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from phenofuncrank_repro.interpretation.schemas.messages import MessageEnvelope, ArtifactRef
from phenofuncrank_repro.interpretation.schemas.plan import ExecutionPlan, PlanStep, TaskType, AgentName, OnFailure
from phenofuncrank_repro.interpretation.logging_utils import get_logger
from phenofuncrank_repro.interpretation.llm_utils import complete_json, configured_api_key, configured_model, now_stamp
from phenofuncrank_repro.interpretation.settings import settings

logger = get_logger(__name__)


NextNode = Literal[
    "vcf_execution",
    "model_execution",
    "knowledge_execution",
    "integration_execution",
    "render",
    "request_input",
    "fail",
]


class Supervisor:
    """Top-level controller that transforms user intent into a constrained plan.

    Uses DeepSeek LLM for intent parsing and plan generation.
    Falls back to deterministic rules when LLM is unavailable.
    """


    TASK_AGENT: dict[TaskType, AgentName] = {
        "validate_request": "supervisor",
        "standardize_vcf": "vcf_agent",
        "run_phenotype_prediction": "model_agent",
        "run_allele_swap": "model_agent",
        "run_cadd": "model_agent",
        "compute_ld": "model_agent",
        "compute_population_metrics": "model_agent",
        "retrieve_knowledge": "evidence_retrieval",
        "integrate_evidence": "hypothesis_reasoning",
        "render_report": "report_agent",
    }

    def __init__(self):
        self._llm_client: Any = None
        self._llm_model: str = ""


    def _init_llm(self) -> bool:
        """Lazy-init DeepSeek client. Returns True on success."""
        if self._llm_client is not None:
            return True
        try:
            from openai import OpenAI
            api_key = configured_api_key()
            if not api_key:
                logger.warning("Supervisor LLM: no API key found")
                return False
            base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            self._llm_model = configured_model()
            self._llm_client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=settings.llm_timeout_seconds,
                max_retries=0,
            )
            logger.info(f"Supervisor LLM: connected to {base_url} model={self._llm_model}")
            return True
        except Exception as exc:
            logger.warning(f"Supervisor LLM init failed: {exc}")
            return False

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str | None:
        """Call DeepSeek and return content string, or None on failure."""
        if not self._init_llm():
            return None
        try:
            response = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{now_stamp()}\n{user_prompt}"},
                ],
                temperature=0.2,
                max_tokens=settings.agent_max_output_tokens,
            )
            content = response.choices[0].message.content or ""
            logger.info(f"Supervisor LLM: response {len(content)} chars")
            return content
        except Exception as exc:
            logger.warning(f"Supervisor LLM call failed: {exc}")
            return None

    @staticmethod
    def plan(
        request: dict[str, Any],
        vcf_result: dict[str, Any] | None = None,
        model_result: dict[str, Any] | None = None,
    ) -> tuple[ExecutionPlan, NextNode]:
        """Generate an ExecutionPlan and determine the first node to visit.

        The logic is deterministic for Round 2.  When the LLM integration
        is ready, this will become a structured-output call.
        """
        request_id = request.get("request_id", uuid.uuid4().hex[:12])
        analyses: list[str] = request.get("requested_analyses", [])
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        steps: list[PlanStep] = []
        now = datetime.now(timezone.utc)


        steps.append(PlanStep(
            step_id="gate0",
            task_type="validate_request",
            assigned_agent="supervisor",
            expected_output_schema="RequestSpec",
            on_failure="abort",
        ))


        analysis_set = set(analyses)

        needs_vcf = bool(
            analysis_set
            & {"allele_swap", "phenotype_prediction", "plant_vimpact",
               "ld_expansion", "cultivar_he", "three_axis"}
        )
        needs_model = bool(
            analysis_set
            & {"allele_swap", "phenotype_prediction", "plant_vimpact",
               "ld_expansion", "cultivar_he", "three_axis"}
        )
        needs_knowledge = bool(
            analysis_set
            & {"functional_annotation", "knowledge_retrieval",
               "literature_retrieval", "biological_integration"}
        )
        needs_integration = bool(
            analysis_set
            & {"biological_integration", "three_axis"}
        )


        if needs_vcf:
            steps.append(PlanStep(
                step_id="standardize_vcf",
                task_type="standardize_vcf",
                assigned_agent="vcf_agent",
                depends_on=["gate0"],
                required_inputs=["input_vcf", "species", "expected_reference"],
                expected_output_schema="VCFStandardizationResult",
                on_failure="abort",
            ))


        if needs_model:
            if "allele_swap" in analysis_set or "three_axis" in analysis_set:
                steps.append(PlanStep(
                    step_id="run_dpc",
                    task_type="run_allele_swap",
                    assigned_agent="model_agent",
                    depends_on=["standardize_vcf"] if needs_vcf else ["gate0"],
                    required_inputs=["normalized_vcf", "species", "trait", "model_id"],
                    expected_output_schema="ModelAgentResult",
                    on_failure="abort",
                ))
            if "phenotype_prediction" in analysis_set:
                steps.append(PlanStep(
                    step_id="run_pheno",
                    task_type="run_phenotype_prediction",
                    assigned_agent="model_agent",
                    depends_on=["standardize_vcf"] if needs_vcf else ["gate0"],
                    required_inputs=["normalized_vcf", "species", "trait"],
                    expected_output_schema="ModelAgentResult",
                    on_failure="abort",
                ))
            if "plant_vimpact" in analysis_set:
                steps.append(PlanStep(
                    step_id="run_cadd",
                    task_type="run_cadd",
                    assigned_agent="model_agent",
                    depends_on=["run_dpc"] if "allele_swap" in analysis_set else [],
                    required_inputs=["candidate_snp_table", "species", "reference"],
                    expected_output_schema="ModelAgentResult",
                    on_failure="degrade",
                    retry_policy="default",
                ))
            if "ld_expansion" in analysis_set:
                steps.append(PlanStep(
                    step_id="compute_ld",
                    task_type="compute_ld",
                    assigned_agent="model_agent",
                    depends_on=["run_dpc"],
                    required_inputs=["normalized_vcf", "core_snp_list"],
                    expected_output_schema="ModelAgentResult",
                    on_failure="degrade",
                ))
            if "cultivar_he" in analysis_set:
                steps.append(PlanStep(
                    step_id="compute_he",
                    task_type="compute_population_metrics",
                    assigned_agent="model_agent",
                    depends_on=["run_dpc"],
                    required_inputs=["normalized_vcf", "sample_groups"],
                    expected_output_schema="ModelAgentResult",
                    on_failure="degrade",
                ))


        if needs_knowledge:
            depends = []
            if needs_model:
                depends.append("run_cadd" if "plant_vimpact" in analysis_set else "run_dpc")
            steps.append(PlanStep(
                step_id="retrieve_knowledge",
                task_type="retrieve_knowledge",
                assigned_agent="evidence_retrieval",
                depends_on=depends,
                required_inputs=["candidate_snp_table", "species", "trait"],
                expected_output_schema="EvidenceRetrievalAgentResult",
                on_failure="degrade",
                retry_policy="default",
            ))


        if needs_integration:
            depends = []
            if needs_knowledge:
                depends.append("retrieve_knowledge")
            elif needs_model:
                depends.append("run_cadd" if "plant_vimpact" in analysis_set else "run_dpc")
            steps.append(PlanStep(
                step_id="integrate_evidence",
                task_type="integrate_evidence",
                assigned_agent="hypothesis_reasoning",
                depends_on=depends,
                required_inputs=["evidence_state"],
                expected_output_schema="IntegrationResult",
                on_failure="abort",
            ))


        steps.append(PlanStep(
            step_id="render",
            task_type="render_report",
            assigned_agent="report_agent",
            depends_on=[
                s.step_id for s in steps
                if s.task_type in ("integrate_evidence", "retrieve_knowledge", "run_dpc", "run_pheno")
            ],
            required_inputs=[],
            expected_output_schema="FinalOutput",
            on_failure="abort",
        ))


        next_node = _determine_next_node(steps)

        plan = ExecutionPlan(
            plan_id=plan_id,
            request_id=request_id,
            steps=steps,
            terminal_condition="all_required_steps_completed",
        )
        return plan, next_node

    @classmethod
    def llm_plan(
        cls,
        request: dict[str, Any],
        vcf_result: dict[str, Any] | None = None,
        model_result: dict[str, Any] | None = None,
    ) -> tuple[ExecutionPlan, NextNode]:
        """LLM-powered planning: interpret user intent and generate ExecutionPlan.

        Falls back to static plan() when LLM is unavailable.
        """
        supervisor = cls()
        if not supervisor._init_llm():
            logger.info("Supervisor LLM unavailable → falling back to static plan()")
            return cls.plan(request, vcf_result, model_result)

        species = request.get("species", "unknown")
        trait = request.get("trait", "unknown")
        goal = request.get("target_direction", "")
        analyses = request.get("requested_analyses", [])
        analysis_type = request.get("analysis_type", "standard")

        system_prompt = (
            "你是一个作物育种分析系统的调度器(Supervisor Agent)。"
            "你的任务是根据用户的分析请求，验证其合理性，并返回JSON格式的分析计划决策。\n\n"
            "支持的物种: rice(水稻), maize(玉米), cotton(棉花), tomato(番茄), chickpea(鹰嘴豆)\n"
            "支持的性状: Plant_height(株高), yield(产量), flowering_time(开花时间) 等\n"
            "支持的分析步骤: allele_swap(DPC效应), phenotype_prediction(表型预测), "
            "plant_vimpact(PlantVImpact功能评分), ld_expansion(LD扩展), cultivar_he(群体遗传), "
            "functional_annotation(功能注释), knowledge_retrieval(知识检索), "
            "literature_retrieval(文献检索), "
            "biological_integration(证据整合)\n\n"
            "返回严格JSON，格式如下，不要包含其他文字:\n"
            '{"valid": true, "species": "rice", "trait": "Plant_height", '
            '"analyses": ["allele_swap","plant_vimpact","ld_expansion","cultivar_he"], '
            '"reasoning": "简要说明", "warnings": []}'
        )

        user_prompt = (
            f"用户请求分析:\n"
            f"- 物种: {species}\n"
            f"- 性状: {trait}\n"
            f"- 育种目标: {goal or '未指定'}\n"
            f"- 分析类型: {analysis_type}\n"
            f"- 请求分析项: {analyses}\n"
            f"- VCF已标准化: {'是' if vcf_result else '否'}\n"
            f"- 模型已运行: {'是' if model_result else '否'}\n\n"
            f"请验证此请求是否合理，并返回分析计划JSON。"
        )

        parsed = complete_json(
            system_prompt,
            {"request": user_prompt},
            max_tokens=settings.llm_synthesis_max_tokens,
        )
        if not parsed:
            logger.info("Supervisor LLM call failed → falling back to static plan()")
            return cls.plan(request, vcf_result, model_result)
        if "analyses" not in parsed:
            logger.warning("Supervisor LLM JSON missing 'analyses' key")
            return cls.plan(request, vcf_result, model_result)


        allowed_analyses = {
            "allele_swap", "phenotype_prediction", "plant_vimpact", "ld_expansion",
            "cultivar_he", "three_axis", "functional_annotation",
            "knowledge_retrieval", "literature_retrieval", "biological_integration",
        }
        validated_analyses = [item for item in analyses if item in allowed_analyses]
        if not validated_analyses:
            validated_analyses = analyses
        warnings_list = parsed.get("warnings", [])
        if not isinstance(warnings_list, list):
            warnings_list = []
        llm_selected = {
            item for item in parsed.get("analyses", []) if item in allowed_analyses
        }
        dropped_by_llm = [item for item in validated_analyses if item not in llm_selected]
        if dropped_by_llm:
            warnings_list.append(
                "Supervisor LLM validation did not include explicitly requested "
                f"analysis items; preserved them: {dropped_by_llm}"
            )
        reasoning = parsed.get("reasoning", "")

        logger.info(f"Supervisor LLM plan: species={parsed.get('species',species)} "
                    f"trait={parsed.get('trait',trait)} analyses={validated_analyses} "
                    f"reasoning={reasoning[:80]}")

        augmented_request = dict(request)
        augmented_request["requested_analyses"] = validated_analyses
        augmented_request["_llm_reasoning"] = reasoning
        augmented_request["_llm_warnings"] = warnings_list

        plan, next_node = cls.plan(augmented_request, vcf_result, model_result)

        plan.plan_id = f"llm_{plan.plan_id}"
        return plan, next_node

    @staticmethod
    def build_vcf_agent_input(
        state: dict[str, Any],
    ) -> MessageEnvelope:
        """Project State into a minimal input for the VCF Agent."""
        req = state["request"]
        return MessageEnvelope(
            message_id=_msg_id(),
            trace_id=req.get("request_id", ""),
            run_id=state.get("control", {}).get("current_task_id", ""),
            task_id="standardize_vcf",
            sender="supervisor",
            recipient="tool_router",
            message_type="STANDARDIZE_VCF_COMMAND",
            schema_version="1.0",
            idempotency_key=_idem("vcf", req),
            status="success",
            payload={
                "species": req.get("species", ""),
                "trait": req.get("trait", ""),
                "expected_reference": req.get("reference_assembly", ""),
                "input_artifact_ids": req.get("input_artifact_ids", []),
                "input_sha256": req.get("input_sha256", ""),
                "required_checks": [
                    "compression", "tabix_index", "contig_names",
                    "reference_allele", "duplicate_variants",
                    "multiallelic_normalization", "sample_names",
                    "missingness", "model_position_coverage",
                ],
            },
        )

    @staticmethod
    def build_model_agent_input(
        state: dict[str, Any],
    ) -> MessageEnvelope:
        """Project State into a minimal input for the Model Agent."""
        req = state["request"]
        vcf = state.get("vcf_result", {})
        species = req.get("species", "")


        

        input_artifacts = req.get("input_artifact_ids", [])
        vcf_path = str(input_artifacts[0]) if input_artifacts else ""

        return MessageEnvelope(
            message_id=_msg_id(),
            trace_id=req.get("request_id", ""),
            run_id=state.get("control", {}).get("current_task_id", ""),
            task_id="model_execution",
            sender="supervisor",
            recipient="tool_router",
            message_type="RUN_DPCFORMER_COMMAND",
            schema_version="1.0",
            idempotency_key=_idem("model", req),
            status="success",
            payload={
                "species": species,
                "trait": req.get("trait", ""),
                "reference": req.get("reference_assembly", ""),
                "requested_analyses": req.get("requested_analyses", []),
                "target_direction": req.get("target_direction"),
                "population_label": req.get("population_label", ""),
                "top_n": req.get("top_n", 20),
                "vcf_path": vcf_path,
                "input_sha256": req.get("input_sha256", ""),
                "normalized_vcf_artifact_id": (
                    (vcf.get("normalized_vcf") or {}).get("uri", "")
                    if vcf else ""
                ),
                "vcf_result": vcf if vcf else {},
                "mode": (
                    "allele_swap"
                    if "allele_swap" in req.get("requested_analyses", [])
                    else "phenotype_prediction"
                ),
                "model_id": _model_for(req.get("species", ""), req.get("trait", "")),
                "minimum_supported_site_fraction": 0.95,
            },
        )

    @staticmethod
    def build_evidence_retrieval_input(
        state: dict[str, Any],
        *,
        candidate_ref_override: dict[str, Any] | None = None,
        graph_plans: dict[str, Any] | None = None,
        run_id_override: str | None = None,
    ) -> MessageEnvelope:
        """Project State into a minimal input for the Knowledge Agent."""
        req = state["request"]
        model = state.get("model_result") or {}
        candidate_ref = candidate_ref_override or model.get("top_candidate_artifact")
        planned_graph = (
            graph_plans
            if graph_plans is not None
            else state.get("graph_plans") or {}
        )

        return MessageEnvelope(
            message_id=_msg_id(),
            trace_id=req.get("request_id", ""),
            run_id=(
                run_id_override
                or state.get("control", {}).get("current_task_id", "")
            ),
            task_id="retrieve_knowledge",
            sender="supervisor",
            recipient="evidence_retrieval",
            message_type="RETRIEVE_KNOWLEDGE_COMMAND",
            schema_version="1.0",
            idempotency_key=_idem("knowledge", req),
            status="success",
        payload={
            "species": req.get("species", ""),
            "trait": req.get("trait", ""),
            "reference_assembly": req.get("reference_assembly", ""),
            "top_candidate_artifact": candidate_ref if isinstance(candidate_ref, dict) else None,
            "annotation_version": "latest",
            "allowed_databases": _knowledge_allowed_databases(req),
            "query_budget": 60,
            "evidence_gaps": state.get("evidence_gaps") or [],
            "prior_evidence_state": state.get("evidence_state") or {},
            "prior_admitted_evidence": (state.get("knowledge_result") or {}).get("trusted_evidence_items") or [],
            "prior_literature_records": ((state.get("knowledge_result") or {}).get("literature_records") or [])[:500],

            
            "graph_plans": planned_graph,
            "graph_plan_owner": "supervisor",
            "runtime_context": state.get("runtime_context"),
        },
    )

    @staticmethod
    def build_graph_planning_context(
        state: dict[str, Any],
        *,
        candidate_ref: dict[str, Any] | None = None,
        max_candidates: int | None = None,
    ) -> dict[str, Any]:
        "Query GraphRAG on behalf of A1 and return planning context."
        req = state.get("request") or {}
        model = state.get("model_result") or {}
        ref = candidate_ref or model.get("top_candidate_artifact")
        if not isinstance(ref, dict):
            return {
                "plans": {},
                "candidate_ids": [],
                "candidate_count": 0,
                "planned_count": 0,
                "errors": ["no candidate artifact supplied"],
            }

        uri = str(ref.get("uri", "")).strip()
        path = Path(uri)
        if not path.is_file():
            return {
                "plans": {},
                "candidate_ids": [],
                "candidate_count": 0,
                "planned_count": 0,
                "errors": [f"candidate artifact not found: {uri}"],
            }

        try:
            import pandas as pd

            frame = pd.read_csv(path, sep="\t")
        except Exception as exc:
            return {
                "plans": {},
                "candidate_ids": [],
                "candidate_count": 0,
                "planned_count": 0,
                "errors": [f"candidate artifact unreadable: {exc}"],
            }

        limit = max_candidates
        if limit is None:
            limit = int(getattr(settings, "parallel_evidence_prefetch_max_candidates", 100))
        limit = max(1, int(limit))
        frame = frame.head(limit)

        candidate_ids = []
        for column in ("user_snp_id", "snp_id", "variant_id"):
            if column in frame.columns:
                candidate_ids = [
                    str(value).strip()
                    for value in frame[column].dropna().tolist()
                    if str(value).strip()
                ]
                break
        if not settings.enable_graph_guided_retrieval:
            return {
                "plans": {},
                "candidate_ids": list(dict.fromkeys(candidate_ids)),
                "candidate_count": len(set(candidate_ids)),
                "planned_count": 0,
                "errors": ["GraphRAG planning disabled by settings"],
            }

        species = str(req.get("species", "")).strip()
        trait = str(req.get("trait", "")).strip()
        vcf_result = state.get("vcf_result") or {}
        assembly = str(
            req.get("reference_assembly")
            or vcf_result.get("detected_reference")
            or vcf_result.get("expected_reference")
            or "unknown"
        ).strip()
        if not assembly or assembly.lower() == "nan":
            assembly = "unknown"


        
        from phenofuncrank_repro.interpretation.graph_rag import GraphRAGPlanner

        planner = GraphRAGPlanner()
        plans: dict[str, dict[str, Any]] = {}
        candidate_ids: list[str] = []
        errors: list[str] = []
        for row in frame.to_dict(orient="records"):
            def _text(*names: str) -> str:
                for name in names:
                    value = row.get(name)
                    if value is not None and str(value).strip().lower() != "nan":
                        text = str(value).strip()
                        if name in {"pos", "position"}:
                            try:
                                numeric = float(text)
                                if numeric.is_integer():
                                    text = str(int(numeric))
                            except ValueError:
                                pass
                        return text
                return ""

            snp_id = _text("user_snp_id", "snp_id", "variant_id")
            chrom = _text("chrom", "chr", "chromosome")
            pos = _text("pos", "position")
            ref_base = _text("ref", "REF")
            alt_base = _text("alt", "ALT")
            if not snp_id:
                snp_id = f"{chrom}:{pos}:{ref_base}>{alt_base}"
            if not snp_id:
                continue
            if snp_id not in candidate_ids:
                candidate_ids.append(snp_id)
            if not (chrom and pos and ref_base and alt_base):
                errors.append(f"{snp_id}: incomplete canonical coordinates")
                continue
            snp_key = "|".join((
                species,
                assembly,
                chrom,
                pos,
                ref_base.upper(),
                alt_base.upper(),
            ))
            try:
                plans[snp_id] = planner.plan(
                    snp_key=snp_key,
                    region=_text("region_type", "region"),
                    consequence=_text("consequence"),
                    gene_name=_text("candidate_gene", "gene", "gene_name"),
                    trait=trait,
                )
            except Exception as exc:
                errors.append(f"{snp_id}: GraphRAG planning failed: {exc}")

        return {
            "plans": plans,
            "candidate_ids": candidate_ids,
            "candidate_count": len(candidate_ids),
            "planned_count": len(plans),
            "errors": errors,
        }

    @staticmethod
    def candidate_ids_from_artifact(candidate_ref: dict[str, Any] | None) -> set[str]:
        """Read stable candidate IDs for safe prefetch coverage checks."""
        if not isinstance(candidate_ref, dict):
            return set()
        path = Path(str(candidate_ref.get("uri", "")))
        if not path.is_file():
            return set()
        try:
            import pandas as pd

            frame = pd.read_csv(path, sep="\t")
        except Exception:
            return set()
        for column in ("user_snp_id", "snp_id", "variant_id"):
            if column in frame.columns:
                return {
                    str(value).strip()
                    for value in frame[column].dropna().tolist()
                    if str(value).strip()
                }
        return set()


    @staticmethod
    def build_hypothesis_reasoning_input(
        state: dict[str, Any],
    ) -> MessageEnvelope:
        """Project State into a minimal input for the Integration Agent."""
        req = state["request"]
        return MessageEnvelope(
            message_id=_msg_id(),
            trace_id=req.get("request_id", ""),
            run_id=state.get("control", {}).get("current_task_id", ""),
            task_id="integrate_evidence",
            sender="supervisor",
            recipient="hypothesis_reasoning",
            message_type="INTEGRATE_EVIDENCE_COMMAND",
            schema_version="1.0",
            idempotency_key=_idem("integration", req),
            status="success",
            payload={
                "species": req.get("species", ""),
                "trait": req.get("trait", ""),
                "target_direction": req.get("target_direction"),
                "evidence_state": state.get("evidence_state") or {},

                

                "revision_feedback": state.get("revision_feedback") or {},
                "runtime_context": state.get("runtime_context"),
            },
        )

    @staticmethod
    def build_critic_agent_input(
        state: dict[str, Any],
    ) -> MessageEnvelope:
        req = state["request"]
        return MessageEnvelope(
            message_id=_msg_id(),
            trace_id=req.get("request_id", ""),
            run_id=state.get("control", {}).get("current_task_id", ""),
            task_id="critique_hypotheses",
            sender="supervisor",
            recipient="evidence_critic",
            message_type="CRITIQUE_HYPOTHESES_COMMAND",
            schema_version="1.0",
            idempotency_key=_idem("critic", req),
            status="success",
            payload={
                "species": req.get("species", ""),
                "trait": req.get("trait", ""),
                "target_direction": req.get("target_direction"),
                "integration_result": state.get("integration_result"),
                "evidence_state": state.get("evidence_state") or {},
                "runtime_context": state.get("runtime_context"),
            },
        )


def _knowledge_allowed_databases(req: dict[str, Any]) -> list[str]:
    """Map requested analyses to the Knowledge Agent's enabled tool scope.

    功能注释 → annotation tools only; 文献查询 → literature tools only;
    both/full → both sets.  This prevents literature-only runs from silently
    executing the full functional-annotation pipeline.
    """
    analyses = set(req.get("requested_analyses", []))
    annotation_on = "functional_annotation" in analyses
    literature_on = "literature_retrieval" in analyses
    dbs: list[str] = []
    if annotation_on:
        dbs += [
            "swissprot", "pfam", "interpro", "plantregmap", "jaspar",
            "blast", "blastn_context", "fimo", "fimo_motif_scan",
            "rnafold", "rnafold_structure_prediction",
        ]
    if literature_on:
        dbs += ["europepmc", "openalex"]
    if not dbs:
        dbs = [
            "swissprot", "pfam", "interpro", "plantregmap", "jaspar",
            "blast", "blastn_context", "fimo", "fimo_motif_scan",
            "rnafold", "rnafold_structure_prediction",
            "europepmc", "openalex",
        ]
    return dbs


def _msg_id() -> str:
    return f"msg_{uuid.uuid4().hex[:12]}"


def _idem(prefix: str, request: dict[str, Any]) -> str:
    rid = request.get("request_id", "unknown")
    return f"{prefix}_{rid}_{uuid.uuid4().hex[:6]}"


def _model_for(species: str, trait: str) -> str:
    return f"dpcformer_{species}_{trait}_best_fold"


def _determine_next_node(steps: list[PlanStep]) -> NextNode:
    """Find the first unexecuted required step and map to graph node."""
    for s in steps:
        if s.task_type == "standardize_vcf":
            return "vcf_execution"
        if s.task_type in ("run_allele_swap", "run_phenotype_prediction",
                           "run_cadd", "compute_ld", "compute_population_metrics"):
            return "model_execution"
        if s.task_type == "retrieve_knowledge":
            return "knowledge_execution"
        if s.task_type == "integrate_evidence":
            return "integration_execution"
        if s.task_type == "render_report":
            return "render"
    return "fail"
