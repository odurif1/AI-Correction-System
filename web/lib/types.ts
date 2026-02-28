// Session types
export interface Session {
  session_id: string;
  status: string;
  created_at: string;
  copies_count: number;
  graded_count: number;
  average_score?: number;
  max_score?: number;
  subject?: string;
  topic?: string;
}

// Pagination types for future backend support
export interface ListSessionsParams {
  offset?: number;
  limit?: number;
  status?: string;
}

export interface PaginatedSessionsResponse {
  sessions: Session[];
  total: number;
  offset: number;
  limit: number;
}

export interface SessionDetail extends Session {
  subject?: string;
  topic?: string;
  copies: CopySummary[];
  graded_copies: GradedCopySummary[];
  question_weights: Record<string, number>;
}

export interface CopySummary {
  id: string;
  student_name?: string;
  page_count: number;
  processed: boolean;
}

export interface GradedCopySummary {
  copy_id: string;
  student_name?: string;
  total_score: number;
  max_score: number;
  confidence: number;
  grades: Record<string, number>;
  max_points_by_question: Record<string, number>;
  grading_audit?: GradingAudit;
  has_disagreements?: boolean;
}

export interface GradingAudit {
  mode: string;
  grading_method: string;
  verification_mode: string;
  providers: ProviderInfo[];
  questions: Record<string, QuestionAudit>;
  student_detection?: StudentDetectionAudit;
  summary: AuditSummary;
}

export interface ProviderInfo {
  id: string;
  model: string;
  tokens?: { prompt: number; completion: number };
}

export interface QuestionAudit {
  llm_results: Record<string, LLMResult>;
  resolution: ResolutionInfo;
}

export interface LLMResult {
  grade: number;
  max_points?: number;
  reading?: string;
  reasoning?: string;
  feedback?: string;
  confidence: number;
}

export interface ResolutionInfo {
  final_grade: number;
  final_max_points: number;
  method: string;
  phases: string[];
  agreement?: boolean;
}

export interface StudentDetectionAudit {
  final_name: string;
  llm_results: Record<string, string>;
  resolution: ResolutionInfo;
}

export interface AuditSummary {
  total_questions: number;
  agreed_initial: number;
  required_verification: number;
  required_ultimatum: number;
  final_agreement_rate: number;
}

export interface CreateSessionRequest {
  subject?: string;
  topic?: string;
  total_questions?: number;
  question_weights?: Record<string, number>;
}

export interface SessionResponse {
  session_id: string;
  status: string;
  created_at: string;
  copies_count: number;
  graded_count: number;
  average_score?: number;
}

// Analytics types
export interface Analytics {
  mean_score: number;
  median_score: number;
  min_score: number;
  max_score: number;
  std_dev: number;
  score_distribution: Record<string, number>;
  question_stats?: Record<string, { mean: number }>;
}

// Disagreement types
export interface Disagreement {
  copy_id: string;
  copy_index: number;
  student_name?: string;
  question_id: string;
  max_points: number;
  llm1: LLMGrade;
  llm2: LLMGrade;
  resolved: boolean;
}

export interface LLMGrade {
  provider: string;
  model: string;
  grade: number;
  confidence: number;
  reasoning: string;
  reading?: string;
}

export interface ResolveDecision {
  action: "llm1" | "llm2" | "average" | "custom";
  custom_grade?: number;
  teacher_guidance?: string;
}

// Progress event types
export type ProgressEventType =
  | "copy_start"
  | "single_pass_start"
  | "single_pass_complete"
  | "analysis_complete"
  | "question_done"
  | "copy_done"
  | "copy_error"
  | "session_complete"
  // Sub-phase events for correction workflow
  | "reading_start"        // Première lecture phase started
  | "reading_done"         // Première lecture phase completed
  | "verification_start"   // Vérification phase started
  | "verification_done"    // Vérification phase completed
  | "ultimatum_start"     // Ultimatum phase started
  | "ultimatum_done";     // Ultimatum phase completed

// Correction sub-phase type
export type CorrectionSubPhase = "reading" | "verification" | "ultimatum";

export interface CorrectionPhaseInfo {
  currentPhase: CorrectionSubPhase;
  phaseLabel: string;  // "Première lecture", "Vérification", "Ultimatum"
}

export interface ProgressEvent {
  type: ProgressEventType;
  [key: string]: unknown;
}

export interface CopyProgress {
  copyIndex: number;
  totalCopies: number;
  copyId?: string;
  studentName?: string;
  status: "pending" | "grading" | "done" | "error";
  questions: QuestionProgress[];
  totalScore?: number;
  maxScore?: number;
  confidence?: number;
  error?: string;
  agreement?: boolean; // For dual-LLM grading agreement
}

export interface QuestionProgress {
  questionId: string;
  grade?: number;
  maxPoints?: number;
  method?: string;
  agreement?: boolean;
  status: "pending" | "grading" | "done";
}

// Provider types
export interface Provider {
  id: string;
  name: string;
  type: "gemini" | "openai" | "openrouter";
  models: ProviderModel[];
  configured: boolean;
}

export interface ProviderModel {
  id: string;
  name: string;
  context_window?: number;
  pricing?: {
    input: number;
    output: number;
  };
}

// Export types
export type ExportFormat = "csv" | "json" | "pdf";

export interface ExportOptions {
  format: ExportFormat;
  include_feedback: boolean;
  include_reasoning: boolean;
  include_analytics: boolean;
}

// Pre-Analysis types
export interface StudentInfo {
  index: number;
  name: string | null;
  start_page: number;
  end_page: number;
  confidence: number;
}

export type DocumentType = "student_copies" | "subject_only" | "random_document" | "unclear";
export type PDFStructure = "one_pdf_one_student" | "one_pdf_all_students" | "ambiguous";
export type SubjectIntegration = "integrated" | "separate" | "not_detected";

export interface PreAnalysisResult {
  analysis_id: string;
  is_valid_pdf: boolean;
  page_count: number;

  // Document type
  document_type: DocumentType;
  confidence_document_type: number;

  // Structure
  structure: PDFStructure;
  subject_integration: SubjectIntegration;
  num_students_detected: number;
  students: StudentInfo[];

  // Grading scale
  grading_scale: Record<string, number>;
  confidence_grading_scale: number;
  questions_detected: string[];

  // Issues
  blocking_issues: string[];
  has_blocking_issues: boolean;
  warnings: string[];
  quality_issues: string[];
  overall_quality_score: number;

  // Metadata
  detected_language: string;
  cached: boolean;
  analysis_duration_ms: number;

  // Exam name (auto-generated by LLM)
  exam_name?: string;
}

export interface PreAnalysisRequest {
  force_refresh?: boolean;
}

export interface ConfirmPreAnalysisRequest {
  confirm: boolean;
  adjustments?: {
    grading_scale?: Record<string, number>;
    students?: StudentInfo[];
    structure?: PDFStructure;
  };
}

export interface ConfirmPreAnalysisResponse {
  success: boolean;
  session_id: string;
  status: string;
  grading_scale: Record<string, number>;
  num_students: number;
}
