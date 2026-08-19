---
title: Attributnutzungsanalyse
description: Messung, welche Attribute tatsächlich befüllt, abgefragt und variiert
  werden, sodass ein generisches Datenmodell durch Evidenz statt durch Vermutungen
  ersetzt werden kann.
category:
- Database
- Code
- Process
problems:
- entity-attribute-value-overuse
- database-schema-design-problems
- excessive-customization
- schema-evolution-paralysis
- slow-database-queries
- difficult-to-understand-code
- high-technical-debt
- invisible-nature-of-technical-debt
- modernization-strategy-paralysis
- inadequate-requirements-gathering
- authorization-role-explosion
- custom-report-sprawl
- low-code-customization-sprawl
layout: solution
lang: de
en_slug: attribute-usage-analysis
related_solutions:
- slug: typed-schema-extraction
  similarity: 0.7
- slug: feature-usage-measurement
  similarity: 0.65
- slug: role-model-rationalization
  similarity: 0.6
- slug: production-like-test-data
  similarity: 0.6
- slug: customization-cost-attribution
  similarity: 0.6
- slug: change-impact-analysis
  similarity: 0.6
---

## Description

Attributnutzungsanalyse stellt aus den Daten selbst fest, welche Attribute in einem generischen Modell tatsächlich genutzt werden: wie viele Entitäten jedes davon befüllt haben, wie viele unterschiedliche Werte jedes annimmt, ob irgendetwas es liest und wann es zuletzt geschrieben wurde. Sie existiert, weil ein generisches Modell seine eigene Struktur verbirgt — das Schema steckt in den Daten, und niemand hat es angeschaut. Die Konsequenz ist, dass jede Diskussion über den Ersatz eines solchen Modells an dem Glauben stagniert, dass die Flexibilität benötigt wird, was niemand bestätigen oder widerlegen kann. Die Analyse findet fast immer dieselbe Form: eine kleine Anzahl von Attributen, die auf fast jeder Entität befüllt sind, welche die tatsächliche Struktur darstellen, die generisch gespeichert wird, ohne verbleibenden Grund, und einen sehr langen Schwanz, der spärlich, veraltet oder tot ist. Diese Verteilung ist es, was eine inkrementelle Korrektur möglich macht.

## How to Apply ◆

> Ein generisches Modell, das seit Jahren in Produktion ist, ist ein Protokoll dessen, was die Organisation tatsächlich brauchte, und es wurde nie als solches gelesen.

- **Beginnen Sie mit Befüllungszählungen.** Für jeden Attributnamen, wie viele Entitäten haben ihn gesetzt, als Anteil an der Gesamtzahl. Diese einzelne Abfrage teilt die Attribute üblicherweise in einen offensichtlichen Kopf und Schwanz und dauert Minuten zur Ausführung.
- **Zählen Sie unterschiedliche Werte pro Attribut.** Ein Attribut mit einem einzigen unterschiedlichen Wert über eine Million Entitäten ist eine Konstante. Eines mit drei ist eine Enumeration, die nie modelliert wurde. Eines mit einer Million ist Freitext. Jedes impliziert ein anderes Zieldesign.
- **Prüfen Sie, wann jedes Attribut zuletzt geschrieben wurde**, wenn das Modell Zeitstempel trägt. Attribute, die seit Jahren nicht geschrieben wurden, sind tote Struktur, und ihre Identifikation ist die günstigste verfügbare Reduktion.
- **Finden Sie heraus, was jedes Attribut liest**, nicht nur was es schreibt. Anwendungscode-Suche, Abfrageprotokolle und Berichtsdefinitionen zusammen ergeben ein nutzbares Bild. Ein Attribut, das von einem Import geschrieben und von nichts gelesen wird, ist ein Kandidat für Löschung statt Migration.
- **Stichprobenprüfen Sie die Werte gegen ihren beabsichtigten Typ.** Zählen Sie, wie viele Einträge in einem Attribut, das numerisch oder ein Datum sein sollte, nicht als solches geparst werden können. Dies produziert den Datenqualitätsbefund, der die Arbeit motiviert, und ist üblicherweise schlimmer, als jemand erwartet.
- **Suchen Sie nach demselben Konzept unter mehreren Namen.** Vokabular driftet in einem Modell, das niemand steuert, und die Konsolidierung von Synonymen ist oft eine große Reduktion scheinbarer Komplexität für sehr wenig Aufwand.
- **Kreuzen Sie die Nutzung mit den Kunden oder Mandanten, die sie befüllen**, wo das Modell mandantenfähig ist. Ein Attribut, das von einem Mandanten genutzt wird, ist eine Anpassung; eines, das von allen genutzt wird, ist Produktstruktur.
- **Veröffentlichen Sie die Verteilung, nicht nur die Schlussfolgerung.** Ein Diagramm, das 31 Attribute über 90 Prozent Befüllung und 700 unter einem Prozent zeigt, ist ein überzeugenderes Argument für Veränderung als jede Beschreibung, und es ist überprüfbar.
- **Führen Sie sie nach jeder Änderung erneut aus**, um zu bestätigen, dass der Schwanz schrumpft, statt durch neuen Schwanz ersetzt zu werden.

## Tradeoffs ⇄

> Die Analyse ist günstig und verwandelt eine unbeweisbare Designdebatte in eine evidenzbasierte, aber Nutzung ist nicht dasselbe wie Wichtigkeit, und die Daten können irreführen.

**Vorteile:**

- Die echte Struktur, die im generischen Modell verborgen ist, wird sichtbar, was die Voraussetzung für den Ersatz jedes Teils davon ist.
- Tote und beinahe tote Attribute werden identifiziert, und ihre Entfernung ist üblicherweise die größte und günstigste verfügbare Reduktion.
- Das Argument darüber, ob die Flexibilität benötigt wird, wird empirisch, was eine Diskussion entblockiert, die sonst auf Behauptung läuft.
- Datenqualitätsprobleme kommen mit einer angehängten Zählung ans Licht, was einen Verdacht in einen Befund verwandelt, auf den jemand reagieren muss.
- Die Unterscheidung mandantenspezifischer von universellen Attributen trennt das Anpassungsproblem vom Datenmodellproblem, die unterschiedliche Antworten brauchen.

**Kosten und Risiken:**

- Geringe Nutzung ist nicht geringe Wichtigkeit. Ein selten befülltes Attribut könnte regulatorisch, vertraglich oder essenziell für einen hochwertigen Kunden sein, und Löschung allein nach Häufigkeit ist gefährlich.
- Attribute, die nur in langen Abständen genutzt werden — jährliche Prozesse, Jahresendberichterstattung — können innerhalb jedes kurzen Beobachtungsfensters tot erscheinen.
- Festzustellen, was ein Attribut liest, ist echt schwierig, wo der Zugriff dynamisch, generiert ist oder von Berichtswerkzeugen außerhalb der Codebasis kommt.
- Die Ausführung der Analyse gegen produktionsgroße Daten kann teuer sein, und gegen eine Stichprobe kann sie genau die seltenen Attribute übersehen, die wichtig sind.
- Die Befunde können genutzt werden, um die Entfernung von Flexibilität zu rechtfertigen, die eine zukünftige Anforderung brauchen wird, und dieses Urteil steckt nicht in den Daten.

## How It Could Be

Ein Team, das ein Auftragsmanagementsystem pflegte, hatte drei Jahre lang darüber diskutiert, ob sein attributbasiertes Produktmodell ersetzt werden könnte. Die Flexibilität wurde als essenziell bezeichnet. Zwei Tage Analyse klärten es. Von 430 verschiedenen Attributnamen waren 24 auf über 95 Prozent der Bestellungen befüllt — dies war die Bestellstruktur, generisch gespeichert seit einer 2013 getroffenen Entscheidung aus Gründen, die niemand rekonstruieren konnte. Weitere 60 waren auf zwischen 1 und 20 Prozent befüllt, konzentriert nach Produktlinie. Die verbleibenden 346 waren unter einem Prozent, und 190 davon waren seit 2020 nicht geschrieben worden. Die Verteilung wurde als einzelnes Diagramm veröffentlicht, und die dreijährige Diskussion endete in einem Meeting.

Die Typ-Stichprobenprüfung produzierte den Befund, der die Arbeit tatsächlich finanziert bekam. Ein Attribut, das einen Geldbetrag hielt, enthielt 1,4 Millionen Werte, von denen etwa 2.600 nicht als Zahl geparst werden konnten: manche hatten Währungssymbole, manche nutzten ein Komma als Dezimaltrennzeichen, und 340 waren leere Zeichenketten. Nachgelagerter Code behandelte die Fehler, indem er den Wert stillschweigend als Null behandelte. Das Team konnte nicht feststellen, wie lange dies bereits geschah oder was es gekostet hatte, und die Unfähigkeit, diese Frage zu beantworten, war selbst der überzeugendste Teil des Berichts.
