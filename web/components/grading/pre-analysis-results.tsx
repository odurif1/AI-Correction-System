"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  AlertCircle,
  CheckCircle2,
  FileText,
  Users,
  Award,
  AlertTriangle,
  Loader2,
  Edit2,
} from "lucide-react";
import type {
  PreAnalysisResult,
  StudentInfo,
} from "@/lib/types";

interface PreAnalysisResultsProps {
  result: PreAnalysisResult;
  onConfirm: (adjustments?: {
    grading_scale?: Record<string, number>;
  }) => void;
  isConfirming?: boolean;
}

export function PreAnalysisResults({
  result,
  onConfirm,
  isConfirming = false,
}: PreAnalysisResultsProps) {
  const [isEditingScale, setIsEditingScale] = useState(false);
  const [editedScale, setEditedScale] = useState<Record<string, number>>(
    result.grading_scale
  );

  const handleScaleChange = (question: string, value: string) => {
    const numValue = parseFloat(value);
    if (!isNaN(numValue) && numValue >= 0) {
      setEditedScale((prev) => ({
        ...prev,
        [question]: numValue,
      }));
    }
  };

  const handleConfirm = () => {
    const hasChanges =
      JSON.stringify(editedScale) !== JSON.stringify(result.grading_scale);
    onConfirm(hasChanges ? { grading_scale: editedScale } : undefined);
  };

  const getDocumentTypeLabel = (type: string) => {
    switch (type) {
      case "student_copies":
        return "Copies d'élèves";
      case "subject_only":
        return "Sujet uniquement";
      case "random_document":
        return "Document non reconnu";
      default:
        return "Type incertain";
    }
  };

  const getStructureLabel = (structure: string) => {
    switch (structure) {
      case "one_pdf_one_student":
        return "1 élève par PDF";
      case "one_pdf_all_students":
        return "Tous les élèves dans ce PDF";
      default:
        return "Structure ambiguë";
    }
  };

  const getSubjectIntegrationLabel = (integration: string) => {
    switch (integration) {
      case "integrated":
        return "Sujet intégré";
      case "separate":
        return "Sujet séparé";
      default:
        return "Sujet non détecté";
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return "text-green-600";
    if (confidence >= 0.5) return "text-yellow-600";
    return "text-red-600";
  };

  return (
    <div className="space-y-4">
      {/* Info banner */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-sm text-blue-800">
        <p className="font-medium mb-1">Vérification du diagnostic</p>
        <p>
          Notre IA a analysé la structure de votre document avant sa correction.
          Vérifiez que le nombre d'élèves et le barème sont corrects.
          Si nécessaire, modifiez les valeurs avant de confirmer.
        </p>
      </div>

      {/* Blocking Issues */}
      {result.has_blocking_issues && (
        <Card className="border-red-200 bg-red-50">
          <CardHeader className="pb-2">
            <CardTitle className="text-lg flex items-center gap-2 text-red-700">
              <AlertCircle className="h-5 w-5" />
              Problèmes bloquants
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-1">
              {result.blocking_issues.map((issue, index) => (
                <li key={index} className="text-red-700 text-sm">
                  {issue}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {/* Document Summary */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-lg flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Document
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4 text-sm">
            {result.exam_name && (
              <div className="col-span-2">
                <span className="text-muted-foreground">Examen:</span>{" "}
                <span className="font-medium text-base">{result.exam_name}</span>
              </div>
            )}
            <div>
              <span className="text-muted-foreground">Pages:</span>{" "}
              <span className="font-medium">{result.page_count}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Type:</span>{" "}
              <Badge variant="outline">{getDocumentTypeLabel(result.document_type)}</Badge>
            </div>
            <div>
              <span className="text-muted-foreground">Structure:</span>{" "}
              <span className="font-medium">{getStructureLabel(result.structure)}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Sujet:</span>{" "}
              <span className="font-medium">
                {getSubjectIntegrationLabel(result.subject_integration)}
              </span>
            </div>
            <div>
              <span className="text-muted-foreground">Confiance:</span>{" "}
              <span className={getConfidenceColor(result.confidence_document_type)}>
                {Math.round(result.confidence_document_type * 100)}%
              </span>
            </div>
            <div>
              <span className="text-muted-foreground">Langue:</span>{" "}
              <span className="font-medium uppercase">{result.detected_language}</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Students Detected */}
      {result.students.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg flex items-center gap-2">
              <Users className="h-5 w-5" />
              Élèves détectés ({result.num_students_detected})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {result.students.map((student) => (
                <StudentRow key={student.index} student={student} />
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Grading Scale */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-lg flex items-center gap-2">
            <Award className="h-5 w-5" />
            Barème détecté
            <Badge
              variant="outline"
              className={getConfidenceColor(result.confidence_grading_scale)}
            >
              {Math.round(result.confidence_grading_scale * 100)}% confiance
            </Badge>
            {!result.has_blocking_issues && (
              <Button
                variant="ghost"
                size="sm"
                className="ml-auto"
                onClick={() => setIsEditingScale(!isEditingScale)}
              >
                <Edit2 className="h-4 w-4 mr-1" />
                {isEditingScale ? "Annuler" : "Modifier"}
              </Button>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {Object.keys(result.grading_scale).length > 0 ? (
            <div className="space-y-2">
              {isEditingScale ? (
                Object.entries(editedScale).map(([question, points]) => (
                  <div key={question} className="flex items-center gap-2">
                    <Label className="w-16">{question}</Label>
                    <Input
                      type="number"
                      step="0.5"
                      min="0"
                      value={points}
                      onChange={(e) => handleScaleChange(question, e.target.value)}
                      className="w-24"
                    />
                    <span className="text-muted-foreground">points</span>
                  </div>
                ))
              ) : (
                <div className="grid grid-cols-3 gap-2">
                  {Object.entries(result.grading_scale).map(
                    ([question, points]) => (
                      <div
                        key={question}
                        className="flex justify-between items-center p-2 bg-muted rounded"
                      >
                        <span className="font-medium">{question}</span>
                        <span>{points} pts</span>
                      </div>
                    )
                  )}
                </div>
              )}
              <div className="mt-2 pt-2 border-t">
                <span className="text-muted-foreground">Total:</span>{" "}
                <span className="font-bold">
                  {Object.values(isEditingScale ? editedScale : result.grading_scale).reduce(
                    (a, b) => a + b,
                    0
                  )}{" "}
                  points
                </span>
              </div>
            </div>
          ) : (
            <p className="text-muted-foreground text-sm">
              Aucun barème détecté. Vous pourrez le définir manuellement.
            </p>
          )}
        </CardContent>
      </Card>

      {/* Warnings */}
      {result.warnings.length > 0 && (
        <Card className="border-yellow-200 bg-yellow-50">
          <CardHeader className="pb-2">
            <CardTitle className="text-lg flex items-center gap-2 text-yellow-700">
              <AlertTriangle className="h-5 w-5" />
              Avertissements
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-1">
              {result.warnings.map((warning, index) => (
                <li key={index} className="text-yellow-700 text-sm">
                  {warning}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {/* Remarks */}
      {result.quality_issues.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-yellow-500" />
              Remarques
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-1">
              {result.quality_issues.map((issue, index) => (
                <li key={index} className="text-muted-foreground text-sm">
                  {issue}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {/* Confirm Button */}
      {!result.has_blocking_issues && (
        <div className="flex justify-end pt-4">
          <Button
            size="lg"
            onClick={handleConfirm}
            disabled={isConfirming}
            className="min-w-[200px]"
          >
            {isConfirming ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Confirmation en cours...
              </>
            ) : (
              <>
                <CheckCircle2 className="h-4 w-4 mr-2" />
                Confirmer et corriger
              </>
            )}
          </Button>
        </div>
      )}

      {/* Cache indicator */}
      {result.cached && (
        <p className="text-xs text-muted-foreground text-center">
          Analyse en cache (durée: {Math.round(result.analysis_duration_ms)}ms)
        </p>
      )}
    </div>
  );
}

function StudentRow({ student }: { student: StudentInfo }) {
  return (
    <div className="flex items-center justify-between p-2 bg-muted rounded">
      <div className="flex items-center gap-2">
        <span className="font-medium">
          {student.name || `Élève ${student.index}`}
        </span>
        <Badge variant="outline" className="text-xs">
          Pages {student.start_page}-{student.end_page}
        </Badge>
      </div>
      <span
        className={`text-xs ${
          student.confidence >= 0.8
            ? "text-green-600"
            : student.confidence >= 0.5
            ? "text-yellow-600"
            : "text-red-600"
        }`}
      >
        {Math.round(student.confidence * 100)}%
      </span>
    </div>
  );
}
